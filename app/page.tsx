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
    if (selectedMethod !== "qr" || !qrCodeRef.current) return;

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

  const handleNumberClick = (num: string) => {
    if (pin.length < 4) setPin(pin + num);
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

  // --- QR Scanner Screen ---
  if (selectedMethod === "qr") {
    return (
      <div className="min-h-screen bg-gradient-to-b from-gray-50 to-white flex flex-col">
        {/* Header */}
        <div className="relative z-20 px-6 pt-6 pb-4 bg-gradient-to-b from-white to-transparent">
          <div className="flex items-center justify-between">
            <button
              onClick={handleCancel}
              className="w-10 h-10 rounded-full bg-gray-200 flex items-center justify-center"
            >
              <svg className="w-5 h-5 text-gray-700" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
            <div className="w-10"></div>
            <div className="w-10"></div>
          </div>
        </div>

        {/* Scanner */}
        <div className="flex-1 flex flex-col items-center justify-center px-6 relative">
          <div className="w-full max-w-sm aspect-square relative rounded-3xl overflow-hidden bg-gray-100 border-2 border-gray-200">
            <div id="qr-scanner" ref={qrCodeRef} className="w-full h-full">
              <video className="w-full h-full object-cover"></video>
            </div>

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

          <div className="text-center mt-8 relative z-20">
            <div className="w-16 h-16 bg-gradient-to-br from-purple-500 to-purple-600 rounded-3xl flex items-center justify-center shadow-xl shadow-purple-500/30 mx-auto mb-4">
              <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
            </div>
            <h1 className="text-xl font-bold text-slate-900 mb-2">
              Uni<span className="text-purple-600">Drop</span>
            </h1>
            <p className="text-gray-500 text-sm">
              {qrResult ? "Authentication successful" : "Position QR code within frame"}
            </p>
          </div>
        </div>

        {/* Footer */}
        <div className="px-6 pb-8 relative z-20 bg-gradient-to-t from-white to-transparent pt-4">
          <div className="flex items-center justify-center gap-2 text-gray-400 text-xs">
            <div className="w-1.5 h-1.5 bg-purple-400 rounded-full"></div>
            <span>SECURED BY UNIDROP</span>
          </div>
        </div>
      </div>
    );
  }

  // --- PIN Entry Screen ---
  if (selectedMethod === "pin") {
    return (
      <div className="min-h-screen bg-gradient-to-b from-slate-50 to-slate-100 flex flex-col">
        {/* Header */}
        <div className="safe-area-top px-6 pt-6 pb-4">
          <div className="flex items-center justify-between">
            <button
              onClick={handleBack}
              className="w-10 h-10 rounded-full bg-slate-200 flex items-center justify-center"
            >
              <svg className="w-5 h-5 text-slate-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
              </svg>
            </button>
            <div className="w-10"></div>
            <div className="w-10"></div>
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 flex flex-col justify-between px-6 py-8">
          <div className="flex-1 flex flex-col justify-center">
            <div className="text-center mb-12">
              <h1 className="text-3xl font-bold text-slate-900 mb-2">
                Uni<span className="text-purple-600">Drop</span>
              </h1>
              <p className="text-slate-400 text-xs tracking-widest uppercase">
                Enter Your PIN
              </p>
            </div>

            <div className="flex justify-center gap-4 mb-16">
              {[0, 1, 2, 3].map((index) => (
                <div
                  key={index}
                  className={`w-3 h-3 rounded-full transition-all duration-200 ${
                    index < pin.length
                      ? "bg-purple-600 scale-110 shadow-lg shadow-purple-500/50"
                      : "bg-slate-200"
                  }`}
                />
              ))}
            </div>

            <div className="max-w-sm mx-auto w-full">
              <div className="grid grid-cols-3 gap-3 mb-6">
                {[1,2,3,4,5,6,7,8,9].map((num) => (
                  <button
                    key={num}
                    onClick={() => handleNumberClick(num.toString())}
                    className="aspect-square rounded-2xl bg-white shadow-sm hover:shadow-md active:scale-95 text-2xl font-semibold text-slate-900 transition-all border border-slate-200"
                  >
                    {num}
                  </button>
                ))}
                <div></div>
                <button
                  onClick={() => handleNumberClick("0")}
                  className="aspect-square rounded-2xl bg-white shadow-sm hover:shadow-md active:scale-95 text-2xl font-semibold text-slate-900 transition-all border border-slate-200"
                >
                  0
                </button>
                <button
                  onClick={handleBackspace}
                  className="aspect-square rounded-2xl bg-white shadow-sm hover:shadow-md active:scale-95 flex items-center justify-center transition-all border border-slate-200"
                >
                  ⌫
                </button>
              </div>
            </div>
          </div>

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
          <h1 className="text-4xl font-bold text-slate-900 mb-2">
            Uni<span className="text-purple-600">Drop</span>
          </h1>
          <p className="text-gray-400 text-xs tracking-widest uppercase">
            Secure Identity Node
          </p>
        </div>

        <div className="space-y-4 max-w-md mx-auto w-full">
          <p className="text-gray-400 text-xs tracking-widest uppercase text-center mb-6">
            Select Entry Method
          </p>

          <button
            onClick={() => setSelectedMethod("qr")}
            className="w-full bg-white rounded-3xl p-8 shadow-lg hover:shadow-xl transition-all duration-300 active:scale-98 border border-gray-100"
          >
            <div className="flex flex-col items-center">
              <div className="w-16 h-16 bg-purple-100 rounded-2xl flex items-center justify-center mb-4">
                <svg className="w-8 h-8 text-purple-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v1m6 11h2m-6 0h-2v4m0-11v3m0 0h.01M12 12h4.01M16 20h4M4 12h4m12 0h.01M5 8h2a1 1 0 001-1V5a1 1 0 00-1-1H5a1 1 0 00-1 1v2a1 1 0 001 1zm12 0h2a1 1 0 001-1V5a1 1 0 00-1-1h-2a1 1 0 00-1 1v2a1 1 0 001 1zM5 20h2a1 1 0 001-1v-2a1 1 0 00-1-1H5a1 1 0 00-1 1v2a1 1 0 001 1z" />
                </svg>
              </div>
              <h3 className="text-lg font-bold text-slate-900 mb-1">
                Scan QR Code
              </h3>
              <p className="text-gray-400 text-sm">
                Use your Digital ID
              </p>
            </div>
          </button>

          <button
            onClick={() => setSelectedMethod("pin")}
            className="w-full bg-white rounded-3xl p-8 shadow-lg hover:shadow-xl transition-all duration-300 active:scale-98 border border-gray-100"
          >
            <div className="flex flex-col items-center">
              <div className="w-16 h-16 bg-purple-100 rounded-2xl flex items-center justify-center mb-4">
                <svg className="w-8 h-8 text-purple-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 20l4-16m2 16l4-16M6 9h14M4 15h14" />
                </svg>
              </div>
              <h3 className="text-lg font-bold text-slate-900 mb-1">
                Enter PIN
              </h3>
              <p className="text-gray-400 text-sm">
                Manual Passcode
              </p>
            </div>
          </button>
        </div>

        <div className="text-center pt-12">
          <div className="inline-flex items-center gap-2 text-gray-400 text-xs tracking-wider uppercase">
            <div className="w-1.5 h-1.5 bg-purple-400 rounded-full"></div>
            <span>UniDrop Cloud Verified</span>
          </div>
        </div>
      </div>
    </div>
  );
}