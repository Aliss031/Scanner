"use client";

import { useState, useEffect, useRef } from "react";
import { Html5Qrcode } from "html5-qrcode";
import { db } from "../lib/firebase";
import { collection, query, where, getDocs, doc, deleteDoc } from "firebase/firestore";

export default function UniDropAuth() {
  const [selectedMethod, setSelectedMethod] = useState<"qr" | "pin" | null>(null);
  const [qrResult, setQrResult] = useState<string | null>(null);
  const [pin, setPin] = useState("");
  const [isVerifying, setIsVerifying] = useState(false);
  const [verificationError, setVerificationError] = useState<string | null>(null);
  const [verifiedData, setVerifiedData] = useState<any>(null);
  const [parcelDocId, setParcelDocId] = useState<string | null>(null);
  const qrCodeRef = useRef<HTMLDivElement | null>(null);
  const html5QrCodeRef = useRef<Html5Qrcode | null>(null);

  // Verify QR Code against database
  const verifyQRCode = async (qrData: string) => {
    setIsVerifying(true);
    setVerificationError(null);

    try {
      // Parse QR data format: "UNIDROP:VfcXa2lmdyV9DycfySC0:596710"
      const qrParts = qrData.split(":");
      if (qrParts.length !== 3 || qrParts[0] !== "UNIDROP") {
        throw new Error("Invalid QR code format");
      }

      const parcelId = qrParts[1];
      const qrPin = qrParts[2];

      // Query Firestore
      const parcelsRef = collection(db, "parcels");
      const q = query(parcelsRef, where("id", "==", parcelId));
      const querySnapshot = await getDocs(q);

      if (querySnapshot.empty) {
        throw new Error("Parcel not found");
      }

      const parcelData = querySnapshot.docs[0].data();
      const docId = querySnapshot.docs[0].id;

      // Verify PIN matches
      if (parcelData.pin !== qrPin) {
        throw new Error("Invalid authentication");
      }

      // Check if already collected
      if (parcelData.status === "COLLECTED") {
        throw new Error("This parcel has already been collected");
      }

      // Check if parcel is scanned (ready to collect)
      if (parcelData.status !== "TO COLLECT") {
        throw new Error("Parcel not ready for collection");
      }

      setVerifiedData(parcelData);
      setParcelDocId(docId);
      
      // Automatically delete parcel after verification
      setTimeout(async () => {
        const parcelRef = doc(db, "parcels", docId);
        await deleteDoc(parcelRef);
      }, 3000);
      
      return true;
    } catch (error: any) {
      setVerificationError(error.message);
      return false;
    } finally {
      setIsVerifying(false);
    }
  };

  // Verify PIN entry against database
  const verifyPIN = async (enteredPin: string) => {
    setIsVerifying(true);
    setVerificationError(null);

    try {
      // Query Firestore for matching PIN
      const parcelsRef = collection(db, "parcels");
      const q = query(parcelsRef, where("pin", "==", enteredPin));
      const querySnapshot = await getDocs(q);

      if (querySnapshot.empty) {
        throw new Error("Invalid PIN");
      }

      const parcelData = querySnapshot.docs[0].data();
      const docId = querySnapshot.docs[0].id;

      // Check if already collected
      if (parcelData.status === "COLLECTED") {
        throw new Error("This parcel has already been collected");
      }

      // Check if parcel is scanned (ready to collect)
      if (parcelData.status !== "TO COLLECT") {
        throw new Error("Parcel not ready for collection");
      }

      setVerifiedData(parcelData);
      setParcelDocId(docId);
      
      // Automatically delete parcel after verification
      setTimeout(async () => {
        const parcelRef = doc(db, "parcels", docId);
        await deleteDoc(parcelRef);
      }, 3000);
      
      return true;
    } catch (error: any) {
      setVerificationError(error.message);
      return false;
    } finally {
      setIsVerifying(false);
    }
  };

  // --- QR Scanner ---
  useEffect(() => {
    initOCR();
    startCamera();
    loadParcels();
    return () => {
      stopCamera();
      workerRef.current?.terminate();
      if (scanIntervalRef.current) clearInterval(scanIntervalRef.current);
    };
  }, []);

    const html5QrCode = new Html5Qrcode("qr-scanner");
    html5QrCodeRef.current = html5QrCode;

    const startScanner = async () => {
      try {
        // ✅ FORCE FRONT CAMERA
        await html5QrCode.start(
          { facingMode: "user" },
          {
            fps: 10,
            qrbox: (viewfinderWidth, viewfinderHeight) => {
              const minEdge = Math.min(viewfinderWidth, viewfinderHeight);
              const qrboxSize = Math.floor(minEdge * 0.8);
              return { width: qrboxSize, height: qrboxSize };
            },
            aspectRatio: 1.0,
          },
          (decodedText) => {
            setQrResult(decodedText);
            html5QrCode.stop();
            verifyQRCode(decodedText);
          },
          () => {}
        );
      } catch {
        // 🔁 FALLBACK → BACK CAMERA
        await html5QrCode.start(
          { facingMode: "environment" },
          { fps: 10 },
          (decodedText) => {
            setQrResult(decodedText);
            html5QrCode.stop();
            verifyQRCode(decodedText);
          },
          () => {}
        );
      }
    };
  }, [ocrReady, isProcessing]);

    startScanner();

    return () => {
      try {
        html5QrCode.stop();
        html5QrCode.clear();
      } catch {}
      html5QrCodeRef.current = null;
    };
  }, [selectedMethod]);

  // Delete parcel from database after successful verification
  const deleteParcelFromDB = async () => {
    if (!parcelDocId) return;

    try {
      const parcelRef = doc(db, "parcels", parcelDocId);
      await deleteDoc(parcelRef);
      
      // Wait a moment before redirecting
      setTimeout(() => {
        handleCancel();
      }, 3000);
    } catch (error: any) {
      console.error("Failed to delete parcel:", error);
    }
  };

  const handleCancel = () => {
    try {
      html5QrCodeRef.current?.stop();
      html5QrCodeRef.current?.clear();
    } catch {}
    setSelectedMethod(null);
    setQrResult(null);
    setVerificationError(null);
    setVerifiedData(null);
    setParcelDocId(null);
  };

  // --- Camera ---
  const startCamera = async () => {
    if (typeof window === "undefined" || !navigator?.mediaDevices) return;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment", width: { ideal: 1920 }, height: { ideal: 1080 } },
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        streamRef.current = stream;
      }
    } catch {
      showNotification("Camera access denied", "error");
    }
  };

  const handleBackspace = () => setPin(pin.slice(0, -1));

  const handleBack = () => {
    setSelectedMethod(null);
    setPin("");
    setVerificationError(null);
    setVerifiedData(null);
    setParcelDocId(null);
  };

  const handleConfirm = async () => {
    await verifyPIN(pin);
  };

  // --- PIN & QR Helpers ---
  const generatePin = () => {
    return Math.floor(100000 + Math.random() * 900000).toString();
  };

  const generateQrData = (parcelId: string, pin: string) => {
    return `UNIDROP:${parcelId}:${pin}`;
  };

            {qrResult && (
              <div className="absolute inset-0 flex items-center justify-center bg-gradient-to-br from-purple-50 to-purple-100 z-20">
                <div className="text-center px-6 w-full">
                  {isVerifying ? (
                    <>
                      <div className="w-16 h-16 border-4 border-purple-600 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
                      <p className="text-slate-900 font-bold text-xl">
                        Verifying...
                      </p>
                    </>
                  ) : verificationError ? (
                    <>
                      <div className="w-16 h-16 bg-red-500 rounded-full flex items-center justify-center mx-auto mb-4 shadow-lg shadow-red-500/30">
                        <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M6 18L18 6M6 6l12 12" />
                        </svg>
                      </div>
                      <p className="text-slate-900 font-bold text-xl mb-2">
                        Verification Failed
                      </p>
                      <p className="text-red-600 text-sm mb-8">
                        {verificationError}
                      </p>
                      <button
                        onClick={handleCancel}
                        className="w-full bg-gradient-to-r from-gray-500 to-gray-600 text-white py-4 rounded-2xl font-bold text-lg shadow-xl active:scale-98 transition-transform"
                      >
                        Try Again
                      </button>
                    </>
                  ) : verifiedData ? (
                    <>
                      <div className="w-16 h-16 bg-green-500 rounded-full flex items-center justify-center mx-auto mb-4 shadow-lg shadow-green-500/30">
                        <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
                        </svg>
                      </div>
                      <p className="text-slate-900 font-bold text-2xl mb-2">
                        Ready for Collection!
                      </p>
                      <div className="bg-white/70 rounded-xl p-6 mb-4 text-left">
                        <p className="text-sm text-gray-600 mb-2">
                          <span className="font-semibold text-slate-900">Tracking:</span> {verifiedData.trackingNumber}
                        </p>
                        <p className="text-sm text-gray-600">
                          <span className="font-semibold text-slate-900">Recipient:</span> {verifiedData.userName}
                        </p>
                      </div>
                      <p className="text-gray-500 text-sm">
                        Your parcel is ready for collection
                      </p>
                    </>
                  ) : null}
                </div>
              </div>
            )}
          </div>

    const pin = generatePin();
    const qrData = generateQrData(parcelRef.id, pin);

    const parcelData: Parcel = {
      id: parcelRef.id,
      trackingNumber,
      status: "TO COLLECT",
      timestamp: Date.now(),
      date: new Date().toLocaleString(),
      userName,
      pin,
      qrData,
    };

    await setDoc(parcelRef, parcelData);
    return parcelData;
  };

  const loadParcels = async () => {
    try {
      const snapshot = await getDocs(collection(db, "parcels"));
      const list: Parcel[] = snapshot.docs.map((d) => d.data() as Parcel);
      setParcels(list.sort((a, b) => b.timestamp - a.timestamp));
    } catch {
      console.warn("Failed to load parcels");
    }
  };

  const deleteParcel = async (id: string) => {
    await deleteDoc(doc(db, "parcels", id));
    showNotification("Parcel deleted", "success");
    loadParcels();
  };

  // --- OCR Scanning ---
  const captureAndScan = async () => {
    if (!videoRef.current || !canvasRef.current || !workerRef.current || isProcessing) return;
    setIsProcessing(true);

          {verificationError && (
            <div className="mb-4 p-4 bg-red-50 border border-red-200 rounded-xl">
              <p className="text-red-600 text-sm text-center font-medium">
                {verificationError}
              </p>
            </div>
          )}

          {verifiedData && (
            <div className="mb-4 p-6 bg-green-50 border border-green-200 rounded-xl">
              <p className="text-green-700 text-base text-center font-bold mb-3">
                ✓ Ready for Collection!
              </p>
              <div className="space-y-1">
                <p className="text-sm text-gray-600 text-center">
                  <span className="font-semibold">Tracking:</span> {verifiedData.trackingNumber}
                </p>
                <p className="text-sm text-gray-600 text-center">
                  <span className="font-semibold">Recipient:</span> {verifiedData.userName}
                </p>
              </div>
            </div>
          )}

          <button
            onClick={handleConfirm}
            disabled={pin.length !== 4 || isVerifying}
            className={`w-full py-4 rounded-2xl text-base font-semibold transition-all ${
              pin.length === 4 && !isVerifying
                ? "bg-gradient-to-r from-purple-600 to-purple-700 text-white shadow-lg shadow-purple-500/30 active:scale-98"
                : "bg-slate-200 text-slate-400 cursor-not-allowed"
            }`}
          >
            {isVerifying ? (
              <span className="flex items-center justify-center gap-2">
                <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                Verifying...
              </span>
            ) : (
              "Confirm PIN"
            )}
          </button>
        </div>
      </div>
    );
  }

  // --- Method Selection Screen ---
  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-50 to-white flex flex-col">
      <div className="flex-1 flex flex-col justify-between px-6 py-12 safe-area">
        <div className="text-center pt-12">
          <div className="inline-flex items-center justify-center w-20 h-20 bg-gradient-to-br from-purple-600 to-purple-700 rounded-3xl mb-6 shadow-lg">
            <svg className="w-10 h-10 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
          </div>
        )}

        <div className="relative bg-black rounded-lg overflow-hidden">
          <video ref={videoRef} autoPlay playsInline muted className="w-full aspect-video object-cover" />
          <canvas ref={canvasRef} className="hidden" />
          {isProcessing && (
            <div className="absolute top-3 left-1/2 -translate-x-1/2 bg-white px-3 py-1 rounded flex items-center gap-2 shadow">
              <Loader className="animate-spin" />
              <span>Scanning for UD#####...</span>
            </div>
          )}
          {scannedText && (
            <div className="absolute bottom-3 left-1/2 -translate-x-1/2 bg-green-600 text-white px-4 py-2 rounded-lg text-lg font-bold shadow">
              {scannedText}
            </div>
          )}
          <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
            <div className="border-4 border-green-400 rounded-xl w-1/2 h-1/3 opacity-70"></div>
          </div>
        </div>

        <div className="bg-white p-4 rounded-lg shadow max-h-80 overflow-y-auto">
          <h2 className="font-bold mb-2 flex items-center gap-2">
            <Database /> Scanned Parcels ({parcels.length})
          </h2>
          {parcels.length === 0 ? (
            <p className="text-gray-600">No parcels scanned yet</p>
          ) : (
            parcels.map((p) => (
              <div key={p.id} className="border-b py-3 space-y-2">
                <p className="font-bold text-green-700">{p.trackingNumber}</p>
                <p className="text-sm text-gray-600">{p.date}</p>
                {p.userName && <p className="text-sm text-blue-700">User: {p.userName}</p>}

                {/* Display PIN */}
                <p className="text-sm font-mono">PIN: {p.pin}</p>

                {/* Display QR code */}
                <QRCodeSVG value={p.qrData} size={100} />

                <button onClick={() => deleteParcel(p.id)}>
                  <Trash2 className="text-red-600 hover:text-red-800" />
                </button>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}
