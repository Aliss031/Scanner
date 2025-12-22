"use client";

import React, { useEffect, useRef, useState } from "react";
import { Database, CheckCircle, XCircle, Loader, Trash2 } from "lucide-react";
import { QRCodeSVG } from "qrcode.react";
import { db } from "../lib/firebase";
import {
  collection,
  doc,
  setDoc,
  getDocs,
  query,
  where,
  deleteDoc,
} from "firebase/firestore";

type Parcel = {
  id: string;
  trackingNumber: string;
  timestamp: number;
  status: string;
  date: string;
  userName?: string;
  pin: string;
  qrData: string;
};

type Notification = {
  message: string;
  type: "success" | "error";
};

export default function ParcelScannerApp() {
  const [isProcessing, setIsProcessing] = useState(false);
  const [scannedText, setScannedText] = useState("");
  const [parcels, setParcels] = useState<Parcel[]>([]);
  const [notification, setNotification] = useState<Notification | null>(null);
  const [ocrReady, setOcrReady] = useState(false);

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const workerRef = useRef<any>(null);
  const scanIntervalRef = useRef<NodeJS.Timeout | null>(null);

  // --- Initialization ---
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

  useEffect(() => {
    if (ocrReady && !scanIntervalRef.current) {
      scanIntervalRef.current = setInterval(() => {
        if (!isProcessing) captureAndScan();
      }, 3000); // scan every 3s
    }
    return () => {
      if (scanIntervalRef.current) {
        clearInterval(scanIntervalRef.current);
        scanIntervalRef.current = null;
      }
    };
  }, [ocrReady, isProcessing]);

  // --- OCR Setup ---
  const initOCR = async () => {
    if (typeof window === "undefined") return;
    try {
      const { createWorker, PSM } = await import("tesseract.js");
      const worker = await createWorker();
      await worker.setParameters({
        tessedit_char_whitelist: "UD0123456789",
        tessedit_pageseg_mode: PSM.SINGLE_LINE,
      });
      workerRef.current = worker;
      setOcrReady(true);
      showNotification("OCR ready for UniDrop scanning", "success");
    } catch (error) {
      console.error("OCR init error:", error);
      showNotification("Failed to initialize OCR", "error");
    }
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

  const stopCamera = () => {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
  };

  // --- Firestore Helpers ---
  const checkUserByUniDropId = async (unidropId: string) => {
    const q = query(collection(db, "users"), where("unidropId", "==", unidropId));
    const snapshot = await getDocs(q);
    if (!snapshot.empty) return snapshot.docs[0].data();
    return null;
  };

  // --- PIN & QR Helpers ---
  const generatePin = () => {
    return Math.floor(100000 + Math.random() * 900000).toString();
  };

  const generateQrData = (parcelId: string, pin: string) => {
    return `UNIDROP:${parcelId}:${pin}`;
  };

  const saveParcelToFirestore = async (trackingNumber: string, userName?: string) => {
    const parcelRef = doc(collection(db, "parcels"));

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

    try {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      const ctx = canvas.getContext("2d")!;
      const w = video.videoWidth;
      const h = video.videoHeight;

      // Crop center region
      const roiX = w * 0.25;
      const roiY = h * 0.35;
      const roiW = w * 0.5;
      const roiH = h * 0.3;
      canvas.width = roiW;
      canvas.height = roiH;
      ctx.drawImage(video, roiX, roiY, roiW, roiH, 0, 0, roiW, roiH);

      // Thresholding
      const imageData = ctx.getImageData(0, 0, roiW, roiH);
      const data = imageData.data;
      for (let i = 0; i < data.length; i += 4) {
        const avg = (data[i] + data[i + 1] + data[i + 2]) / 3;
        const val = avg > 150 ? 255 : 0;
        data[i] = data[i + 1] = data[i + 2] = val;
      }
      ctx.putImageData(imageData, 0, 0);

      // OCR
      const { data: { text } } = await workerRef.current.recognize(canvas);
      const cleanText = text.replace(/[^A-Z0-9]/gi, "").toUpperCase();
      const match = cleanText.match(/UD\d{5}/);

      if (match) {
        const trackingNumber = match[0];
        setScannedText(trackingNumber);

        const user = await checkUserByUniDropId(trackingNumber);
        if (user) {
          await saveParcelToFirestore(trackingNumber, user.fullName);
          showNotification(`✅ Scanned & matched: ${user.fullName}`, "success");
        } else {
          showNotification(`⚠️ UD ID not found in users`, "error");
        }

        loadParcels();
      }
    } catch (error) {
      console.error("OCR error:", error);
    }

    setIsProcessing(false);
  };

  const showNotification = (message: string, type: "success" | "error") => {
    setNotification({ message, type });
    setTimeout(() => setNotification(null), 3000);
  };

  // --- Render ---
  return (
    <div className="min-h-screen bg-gray-100 text-black">
      <div className="max-w-4xl mx-auto p-4 space-y-6">
        <h1 className="text-2xl font-bold">📦 UniDrop Scanner (Firestore)</h1>

        {notification && (
          <div
            className={`p-3 rounded-lg flex items-center gap-2 ${
              notification.type === "success" ? "bg-green-100" : "bg-red-100"
            }`}
          >
            {notification.type === "success" ? <CheckCircle /> : <XCircle />}
            <span className="font-medium">{notification.message}</span>
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