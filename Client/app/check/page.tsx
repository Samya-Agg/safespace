"use client"

import type React from "react"

import { useState, useEffect, useRef } from "react"
import { gsap } from "gsap"
import Link from "next/link"
import { Mic, MicOff, Wifi, WifiOff, Activity, AlertCircle, CheckCircle, X } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Progress } from "@/components/ui/progress"
import { Alert, AlertDescription } from "@/components/ui/alert"
import Results from "./results"

const DASS21_QUESTIONS = [
  "I found it hard to wind down",                  // q1(S)
  "I tended to over-react to situations",          // q6(s)
  "I felt that I was using a lot of nervous energy", // q8(s)
  "I found myself getting agitated",               // q11(s)
  "I found it difficult to relax",                 // q12(s)
  "I was intolerant of anything that kept me from getting on with what I was doing", // q14(s)
  "I felt that I was rather touchy"                // q18(s)
]

export default function CheckPage() {
  const [deviceConnected, setDeviceConnected] = useState(false)
  const [isRecording, setIsRecording] = useState(false)
  const [dass21Responses, setDass21Responses] = useState<number[]>(new Array(7).fill(0))
  const [stressResult, setStressResult] = useState<any>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [uploadedFile, setUploadedFile] = useState<File | null>(null)
  const [audioFile, setAudioFile] = useState<File | null>(null)
  const [audioURL, setAudioURL] = useState<string | null>(null)
  const [recordingTime, setRecordingTime] = useState(0)
  const [allDataReady, setAllDataReady] = useState(false)
  
  // Audio recording refs
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const audioRef = useRef<HTMLAudioElement | null>(null)
  const timerRef = useRef<NodeJS.Timeout | null>(null)

  const pageRef = useRef<HTMLDivElement>(null)
  const resultsRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const ctx = gsap.context(() => {
      gsap.fromTo(
        ".check-section",
        { y: 40, opacity: 0 },
        { y: 0, opacity: 1, duration: 0.8, stagger: 0.15, ease: "power3.out" },
      )
    })

    return () => ctx.revert()
  }, [])

  useEffect(() => {
    if (stressResult && resultsRef.current) {
      gsap.fromTo(
        resultsRef.current,
        { scale: 0.9, opacity: 0, y: 30 },
        { scale: 1, opacity: 1, y: 0, duration: 0.6, ease: "back.out(1.7)" },
      )
    }
  }, [stressResult])

  // Check if required modalities have data (physiological and questionnaire are required, voice is optional)
  useEffect(() => {
    const hasPhysiological = uploadedFile !== null
    const hasQuestionnaire = dass21Responses.some((response) => response > 0)

    setAllDataReady(hasPhysiological && hasQuestionnaire)
  }, [uploadedFile, dass21Responses])

  const handleDass21Change = (index: number, value: number) => {
    const newResponses = [...dass21Responses]
    newResponses[index] = value
    setDass21Responses(newResponses)
  }

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file) {
      setUploadedFile(file)
    }
  }

  const handleAudioUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file) {
      setAudioFile(file)
      setAudioURL(URL.createObjectURL(file))
    }
  }

  const simulateDeviceConnection = () => {
    setDeviceConnected(!deviceConnected)
  }

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          sampleRate: 44100
        } 
      });

      mediaRecorderRef.current = new MediaRecorder(stream, {
        mimeType: 'audio/webm;codecs=opus'
      });

      const chunks: Blob[] = [];
      
      mediaRecorderRef.current.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunks.push(event.data);
        }
      };

      mediaRecorderRef.current.onstop = () => {
        const blob = new Blob(chunks, { type: 'audio/webm' });
        const audioFile = new File([blob], "recorded_audio.webm", { type: "audio/webm" });
        setAudioFile(audioFile);
        setAudioURL(URL.createObjectURL(blob));
        
        // Stop all tracks
        stream.getTracks().forEach(track => track.stop());
      };

      mediaRecorderRef.current.start();
      setIsRecording(true);
      setRecordingTime(0);

      // Start timer
      timerRef.current = setInterval(() => {
        setRecordingTime(prev => prev + 1);
      }, 1000);

    } catch (err) {
      console.error('Error starting recording:', err);
      alert('Failed to access microphone. Please check permissions.');
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      
      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
    }
  };

  const toggleRecording = () => {
    if (isRecording) {
      stopRecording();
    } else {
      startRecording();
    }
  };

//   const analyzeAllModalities = async () => {
//   if (!allDataReady || !uploadedFile || !audioFile) return;

//   setIsLoading(true);
//   try {
//     const formData = new FormData();
//     formData.append("physiological_file", uploadedFile);
//     formData.append("audio_file", audioFile);
//     formData.append("questionnaire", JSON.stringify(dass21Responses));

//     const response = await fetch("http://localhost:8000/api/predict/comprehensive", {
//       method: "POST",
//       body: formData,
//     });

//     const result = await response.json();
//     setStressResult(result);
//   } catch (error) {
//     console.error("Error analyzing comprehensive data:", error);
//   } finally {
//     setIsLoading(false);
//   }
// };

const analyzeAllModalities = async () => {
  if (!allDataReady || !uploadedFile) return;

  setIsLoading(true);
  try {
    // Convert DASS-21 integer responses to a comma-separated string
    const dass21ResponseString = dass21Responses.join(",");

    const formData = new FormData();
    formData.append("physiological_file", uploadedFile);
    formData.append("dass21_responses", dass21ResponseString);

    // Add voice audio if available
    if (audioFile) {
      formData.append("voice_audio", audioFile, audioFile.name);
      console.log("✅ Voice audio file details:", {
        name: audioFile.name,
        type: audioFile.type,
        size: audioFile.size,
        lastModified: audioFile.lastModified
      });
    }

    // Debug prints
    console.log("✅ Sending physiological_file:", uploadedFile.name);
    console.log("✅ DASS-21 Responses:", dass21ResponseString);
    if (audioFile) {
      console.log("✅ Voice audio file:", audioFile.name);
    }

    // Debug: Log FormData contents
    console.log("📋 FormData contents:");
    for (let [key, value] of formData.entries()) {
      if (value instanceof File) {
        console.log(`  ${key}: File(${value.name}, ${value.type}, ${value.size} bytes)`);
      } else {
        console.log(`  ${key}: ${value}`);
      }
    }

    const response = await fetch("http://localhost:8000/predict", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const result = await response.json();
    console.log("✅ API Response:", result);
    setStressResult(result);
  } catch (error) {
    console.error("❌ Error analyzing comprehensive data:", error);
    // You might want to show an error message to the user here
  } finally {
    setIsLoading(false);
  }
};

  const getStressColor = (level: string) => {
    switch (level?.toLowerCase()) {
      case "low":
        return "text-green-600 bg-green-50 border-green-200"
      case "moderate":
        return "text-orange-600 bg-orange-50 border-orange-200"
      case "high":
        return "text-red-600 bg-red-50 border-red-200"
      default:
        return "text-gray-600 bg-gray-50 border-gray-200"
    }
  }

  const getCompletionPercentage = () => {
    let completed = 0
    if (uploadedFile) completed += 40
    if (dass21Responses.some((response) => response > 0)) completed += 40
    if (audioFile) completed += 20
    return completed
  }

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const playAudio = () => {
    if (audioRef.current) {
      audioRef.current.play();
    }
  };

  const clearAudio = () => {
    if (audioURL) {
      URL.revokeObjectURL(audioURL);
    }
    setAudioURL(null);
    setAudioFile(null);
    setRecordingTime(0);
  };

  return (
    <main
      ref={pageRef}
      id="check-page-main"
      className="min-h-screen bg-gradient-to-br from-blue-50 via-teal-50 to-purple-50 py-4 sm:py-6 md:py-8 pt-[120px] scroll-mt-[120px]"
      style={{ scrollMarginTop: '120px' }}
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-8 sm:mb-10 md:mb-12">
          <div className="pt-8 sm:pt-12 md:pt-16"></div>
          <h1 className="text-3xl sm:text-4xl md:text-5xl lg:text-6xl font-bold text-gray-900 mb-3 sm:mb-4 md:mb-6">
            Comprehensive Stress Analysis
          </h1>
          <p className="text-base sm:text-lg md:text-xl text-gray-600 mb-6 sm:mb-8">
            Complete all three assessments for accurate multimodal stress detection
          </p>
          <div className="flex justify-center mb-4">
            <Link
              href="/stress-buster"
              className="inline-flex items-center space-x-2 px-4 py-2 bg-gradient-to-r from-purple-500 to-pink-500 text-white text-sm font-medium rounded-full hover:from-purple-600 hover:to-pink-600 transition-all duration-200 shadow-lg hover:shadow-xl"
            >
              <span>🎮</span>
              <span>Need a break? Try our StressBuster Games!</span>
            </Link>
          </div>

          {/* Progress Indicator */}
          <div className="max-w-md mx-auto mb-6 sm:mb-8">
            <div className="flex justify-between text-sm text-gray-600 mb-2">
              <span>Assessment Progress</span>
              <span>{getCompletionPercentage()}%</span>
            </div>
            <Progress value={getCompletionPercentage()} className="h-2 sm:h-3" />
          </div>

          {!allDataReady && (
            <Alert className="max-w-2xl mx-auto mb-6 sm:mb-8">
              <AlertCircle className="h-4 w-4" />
              <AlertDescription className="text-sm sm:text-base">
                Please complete Physiological Data and Questionnaire for analysis. Voice recording is optional but recommended for enhanced accuracy.
              </AlertDescription>
            </Alert>
          )}
        </div>

        <div className="grid lg:grid-cols-2 gap-6 sm:gap-8">
          {/* Left Column - Input Methods */}
          <div className="space-y-6 sm:space-y-8">
            {/* Hardware Integration Section */}
            <Card className="check-section border-0 shadow-lg sm:shadow-xl bg-white/80 backdrop-blur-sm">
              <CardHeader className="pb-4 sm:pb-6">
                <CardTitle className="flex items-center space-x-2 sm:space-x-3 text-lg sm:text-xl md:text-2xl">
                  <Activity className="w-5 h-5 sm:w-6 sm:h-6 md:w-7 md:h-7 text-blue-600" />
                  <span>Physiological Data</span>
                  {uploadedFile && <CheckCircle className="w-4 h-4 sm:w-5 sm:h-5 text-green-500" />}
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4 sm:space-y-6">
                <div className="flex items-center justify-between p-4 sm:p-6 bg-gradient-to-r from-gray-50 to-blue-50 rounded-xl sm:rounded-2xl border">
                  <div className="flex items-center space-x-3 sm:space-x-4">
                    {deviceConnected ? (
                      <Wifi className="w-5 h-5 sm:w-6 sm:h-6 md:w-7 md:h-7 text-green-600" />
                    ) : (
                      <WifiOff className="w-5 h-5 sm:w-6 sm:h-6 md:w-7 md:h-7 text-gray-400" />
                    )}
                    <span
                      className={`text-sm sm:text-base font-semibold ${deviceConnected ? "text-green-600" : "text-gray-600"}`}
                    >
                      {deviceConnected ? "Device Connected ✅" : "No Device Connected"}
                    </span>
                  </div>
                  <Button
                    onClick={simulateDeviceConnection}
                    variant={deviceConnected ? "destructive" : "default"}
                    size="sm"
                    className="rounded-lg sm:rounded-xl text-xs sm:text-sm"
                  >
                    {deviceConnected ? "Disconnect" : "Connect"}
                  </Button>
                </div>

                <div>
                  <Label
                    htmlFor="file-upload"
                    className="block text-sm sm:text-base md:text-lg font-semibold mb-2 sm:mb-4"
                  >
                    Upload Physiological Data (CSV/JSON)
                  </Label>
                  <div className="flex items-center space-x-2 sm:space-x-4">
                    <Input
                      id="file-upload"
                      type="file"
                      accept=".csv,.json"
                      onChange={handleFileUpload}
                      className="flex-1 p-3 sm:p-4 text-sm sm:text-base md:text-lg rounded-lg sm:rounded-xl"
                    />
                  </div>
                  {uploadedFile && (
                    <p className="text-green-600 mt-2 font-medium text-sm sm:text-base flex items-center space-x-2">
                      <CheckCircle className="w-4 h-4" />
                      <span>{uploadedFile.name} uploaded</span>
                    </p>
                  )}
                </div>
              </CardContent>
            </Card>

            {/* Voice Recording Section */}
            <Card className="check-section border-0 shadow-lg sm:shadow-xl bg-white/80 backdrop-blur-sm">
              <CardHeader className="pb-4 sm:pb-6">
                <CardTitle className="flex items-center space-x-2 sm:space-x-3 text-lg sm:text-xl md:text-2xl">
                  <Mic className="w-5 h-5 sm:w-6 sm:h-6 md:w-7 md:h-7 text-teal-600" />
                  <span>Voice Analysis</span>
                  {(audioFile || isRecording) && <CheckCircle className="w-4 h-4 sm:w-5 sm:h-5 text-green-500" />}
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4 sm:space-y-6">
                {!audioURL ? (
                  <div className="text-center space-y-4 sm:space-y-6">
                    <Button
                      onClick={toggleRecording}
                      className={`w-28 h-28 sm:w-32 sm:h-32 md:w-40 md:h-40 rounded-full text-white font-bold text-base sm:text-lg md:text-xl ${
                        isRecording
                          ? "bg-red-500 hover:bg-red-600 animate-pulse shadow-2xl shadow-red-500/50"
                          : "bg-teal-600 hover:bg-teal-700 shadow-2xl shadow-teal-500/50"
                      } transition-all duration-300 hover:scale-105`}
                    >
                      {isRecording ? (
                        <div className="flex flex-col items-center">
                          <MicOff className="w-8 h-8 sm:w-10 sm:h-10 md:w-16 md:h-16 mb-1 sm:mb-2" />
                          <span className="text-xs sm:text-sm">Stop</span>
                        </div>
                      ) : (
                        <div className="flex flex-col items-center">
                          <Mic className="w-8 h-8 sm:w-10 sm:h-10 md:w-16 md:h-16 mb-1 sm:mb-2" />
                          <span className="text-xs sm:text-sm">Start</span>
                        </div>
                      )}
                    </Button>
                    <p className="text-gray-600 font-medium text-sm sm:text-base">
                      {isRecording ? "Recording your voice... Click to stop" : "Click to start voice recording"}
                    </p>
                    {isRecording && (
                      <div className="text-center text-sm text-gray-600">
                        Recording: {formatTime(recordingTime)}
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="space-y-3">
                    <div className="flex items-center gap-2">
                      <audio
                        ref={audioRef}
                        src={audioURL}
                        controls
                        className="flex-1"
                      />
                      <Button
                        onClick={clearAudio}
                        variant="ghost"
                        size="sm"
                        className="text-red-600 hover:text-red-700"
                      >
                        <X className="h-4 w-4" />
                      </Button>
                    </div>
                    <div className="text-xs text-gray-500">
                      Click play to verify your recording before submitting
                    </div>
                  </div>
                )}

                <div className="border-t pt-4 sm:pt-6">
                  <Label
                    htmlFor="audio-upload"
                    className="block text-sm sm:text-base md:text-lg font-semibold mb-2 sm:mb-4"
                  >
                    Or Upload Audio File
                  </Label>
                  <Input
                    id="audio-upload"
                    type="file"
                    accept=".wav,.mp3,.m4a,.webm"
                    onChange={handleAudioUpload}
                    className="p-3 sm:p-4 text-sm sm:text-base md:text-lg rounded-lg sm:rounded-xl"
                  />
                  {audioFile && audioFile.name !== "recorded_audio.webm" && (
                    <p className="text-green-600 mt-2 font-medium text-sm sm:text-base flex items-center space-x-2">
                      <CheckCircle className="w-4 h-4" />
                      <span>{audioFile.name} uploaded</span>
                    </p>
                  )}
                  {audioFile && audioFile.name === "recorded_audio.webm" && (
                    <p className="text-green-600 mt-2 font-medium text-sm sm:text-base flex items-center space-x-2">
                      <CheckCircle className="w-4 h-4" />
                      <span>Voice recorded successfully</span>
                    </p>
                  )}
                </div>
              </CardContent>
            </Card>

            {/* DASS-21 Questionnaire */}
            <Card className="check-section border-0 shadow-lg sm:shadow-xl bg-white/80 backdrop-blur-sm">
              <CardHeader className="pb-4 sm:pb-6">
                <CardTitle className="flex items-center space-x-2 sm:space-x-3 text-lg sm:text-xl md:text-2xl">
                  <span>DASS-21 Questionnaire</span>
                  {dass21Responses.some((response) => response > 0) && (
                    <CheckCircle className="w-4 h-4 sm:w-5 sm:h-5 text-green-500" />
                  )}
                </CardTitle>
                <p className="text-gray-600 font-medium text-sm sm:text-base">
                  Rate each statement: 0 = Never, 1 = Sometimes, 2 = Often, 3 = Almost Always
                </p>
              </CardHeader>
              <CardContent className="space-y-4 sm:space-y-6 max-h-80 sm:max-h-96 overflow-y-auto">
                {DASS21_QUESTIONS.map((question, index) => (
                  <div key={index} className="space-y-2 sm:space-y-3 p-3 sm:p-4 bg-gray-50 rounded-lg sm:rounded-xl">
                    <Label className="font-medium text-gray-800 text-sm sm:text-base">
                      {index + 1}. {question}
                    </Label>
                    <div className="flex space-x-2 sm:space-x-3">
                      {[0, 1, 2, 3].map((value) => (
                        <Button
                          key={value}
                          variant={dass21Responses[index] === value ? "default" : "outline"}
                          size="sm"
                          onClick={() => handleDass21Change(index, value)}
                          className="w-10 h-8 sm:w-12 sm:h-10 md:w-16 md:h-12 text-sm sm:text-base md:text-lg font-semibold rounded-lg sm:rounded-xl"
                        >
                          {value}
                        </Button>
                      ))}
                    </div>
                  </div>
                ))}
              </CardContent>
            </Card>
          </div>

          {/* Right Column - Analysis & Results */}
          <div className="space-y-6 sm:space-y-8">
            {/* Analysis Button */}
            <Card className="check-section border-0 shadow-lg sm:shadow-xl bg-white/80 backdrop-blur-sm">
              <CardContent className="p-6 sm:p-8 text-center">
                <Button
                  onClick={analyzeAllModalities}
                  disabled={!allDataReady || isLoading}
                  className="w-full py-4 sm:py-6 text-base sm:text-lg md:text-xl font-bold bg-gradient-to-r from-blue-600 via-teal-600 to-purple-600 hover:from-blue-700 hover:via-teal-700 hover:to-purple-700 text-white rounded-xl sm:rounded-2xl shadow-2xl disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-300 hover:scale-105"
                >
                  {isLoading ? (
                    <div className="flex items-center justify-center space-x-3">
                      <div className="animate-spin rounded-full h-5 w-5 sm:h-6 sm:w-6 border-b-2 border-white"></div>
                      <span>Analyzing...</span>
                    </div>
                  ) : (
                    "Analyze Stress Level"
                  )}
                </Button>
                {!allDataReady && (
                  <p className="text-gray-500 mt-3 sm:mt-4 text-sm sm:text-base">
                    Complete Physiological Data and Questionnaire to enable analysis
                  </p>
                )}
              </CardContent>
            </Card>

            {/* Results */}
            <Results result={stressResult} isLoading={isLoading} />


          </div>
        </div>
      </div>
    </main>
  )
}
