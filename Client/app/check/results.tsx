"use client"

import React from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { Badge } from "@/components/ui/badge"
import { 
  Activity, 
  Brain, 
  Mic, 
  FileText, 
  TrendingUp, 
  TrendingDown, 
  AlertTriangle,
  CheckCircle,
  Info,
  BarChart3,
  Lightbulb,
  Clock,
  Zap
} from "lucide-react"

interface StressResult {
  success: boolean
  predictions: {
    physio_probs: number[]
    dass21_probs: number[]
    voice_probs: number[] | null
    fusion_probs: number[]
    fusion_pred: number
    prediction_label: string
    confidence: number
  }
  explanations: {
    physiological: {
      available: boolean
      method: string
      feature_importance: Array<{
        feature: string
        importance: number
        abs_importance: number
      }>
      summary: string
    }
    questionnaire: {
      available: boolean
      method: string
      feature_importance: Array<{
        feature: string
        importance: number
        abs_importance: number
        value: number
      }>
      summary: string
    }
    voice: {
      available: boolean
      method: string
      feature_importance: Array<{
        feature: string
        importance: number
        abs_importance: number
        value: number
      }>
      summary: string
    }
    fusion: {
      available: boolean
      method: string
      modality_contributions: Array<{
        modality: string
        probabilities: number[]
        predicted_class: number
        confidence: number
        entropy: number
        contribution_score: number
      }>
      summary: string
    }
  }
  metadata: {
    physio_windows: number
    physio_features: number
    dass21_values: number[]
    voice_provided: boolean
    modalities_used: string[]
  }
}

interface ResultsProps {
  result: StressResult | null
  isLoading: boolean
}

const Results: React.FC<ResultsProps> = ({ result, isLoading }) => {
  if (isLoading) {
    return (
      <Card className="border-0 shadow-2xl bg-white/90 backdrop-blur-sm">
        <CardContent className="p-8 text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <h3 className="text-xl font-bold text-gray-700 mb-2">Analyzing Your Data</h3>
          <p className="text-gray-500">Processing physiological signals, voice patterns, and questionnaire responses...</p>
        </CardContent>
      </Card>
    )
  }

  if (!result) {
    return (
      <Card className="border-0 shadow-lg bg-white/90 backdrop-blur-sm">
        <CardContent className="p-8 text-center">
          <Activity className="w-16 h-16 text-gray-400 mx-auto mb-4" />
          <h3 className="text-xl font-bold text-gray-600 mb-2">
            Ready for Comprehensive Analysis
          </h3>
          <p className="text-gray-500">
            Complete all three assessment methods to receive your detailed multimodal stress analysis.
          </p>
        </CardContent>
      </Card>
    )
  }

  const getStressLevelColor = (level: string) => {
    switch (level.toLowerCase()) {
      case "low":
        return "text-green-600 bg-green-50 border-green-200"
      case "medium":
        return "text-orange-600 bg-orange-50 border-orange-200"
      case "high":
        return "text-red-600 bg-red-50 border-red-200"
      default:
        return "text-gray-600 bg-gray-50 border-gray-200"
    }
  }

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return "text-green-600"
    if (confidence >= 0.6) return "text-orange-600"
    return "text-red-600"
  }

  const formatFeatureName = (feature: string) => {
    return feature
      .replace(/_/g, " ")
      .replace(/\b\w/g, l => l.toUpperCase())
      .replace("Dass21", "DASS-21")
  }

  return (
    <div className="space-y-6">
      {/* Main Prediction Result */}
      <Card className={`border-2 shadow-2xl ${getStressLevelColor(result.predictions.prediction_label)} bg-white/95 backdrop-blur-sm`}>
        <CardHeader className="text-center pb-4">
          <CardTitle className="text-2xl font-bold flex items-center justify-center gap-2">
            <Zap className="w-6 h-6" />
            Stress Analysis Results
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Primary Result */}
          <div className="text-center">
            <div className="text-6xl font-bold mb-2">
              {result.predictions.prediction_label}
            </div>
            <div className="text-lg text-gray-600 mb-4">
              Stress Level Prediction
            </div>
            <div className="flex items-center justify-center gap-2 mb-4">
              <span className="text-sm text-gray-500">Confidence:</span>
              <Badge className={`${getConfidenceColor(result.predictions.confidence)} bg-white border`}>
                {(result.predictions.confidence * 100).toFixed(1)}%
              </Badge>
            </div>
          </div>

          {/* Fusion Probabilities */}
          <div className="space-y-3">
            <h4 className="font-semibold text-lg flex items-center gap-2">
              <BarChart3 className="w-5 h-5" />
              Prediction Probabilities
            </h4>
            <div className="grid grid-cols-3 gap-3">
              {["Low", "Medium", "High"].map((level, index) => (
                <div key={level} className="text-center p-3 bg-gray-50 rounded-lg">
                  <div className="text-2xl font-bold text-gray-700">
                    {(result.predictions.fusion_probs[index] * 100).toFixed(1)}%
                  </div>
                  <div className="text-sm text-gray-500">{level}</div>
                  <Progress 
                    value={result.predictions.fusion_probs[index] * 100} 
                    className="mt-2 h-2"
                  />
                </div>
              ))}
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Individual Modality Results */}
      <div className="grid md:grid-cols-3 gap-4">
        {/* Physiological Results */}
        <Card className="border-0 shadow-lg bg-white/90 backdrop-blur-sm">
          <CardHeader className="pb-3">
            <CardTitle className="text-lg flex items-center gap-2">
              <Activity className="w-5 h-5 text-blue-600" />
              Physiological
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="space-y-2">
              {["Low", "Medium", "High"].map((level, index) => (
                <div key={level} className="flex justify-between items-center">
                  <span className="text-sm text-gray-600">{level}</span>
                  <span className="text-sm font-semibold">
                    {(result.predictions.physio_probs[index] * 100).toFixed(1)}%
                  </span>
                </div>
              ))}
            </div>
            <div className="text-xs text-gray-500">
              {result.metadata.physio_windows} windows analyzed
            </div>
          </CardContent>
        </Card>

        {/* DASS-21 Results */}
        <Card className="border-0 shadow-lg bg-white/90 backdrop-blur-sm">
          <CardHeader className="pb-3">
            <CardTitle className="text-lg flex items-center gap-2">
              <FileText className="w-5 h-5 text-purple-600" />
              Questionnaire
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="space-y-2">
              {["Low", "Medium", "High"].map((level, index) => (
                <div key={level} className="flex justify-between items-center">
                  <span className="text-sm text-gray-600">{level}</span>
                  <span className="text-sm font-semibold">
                    {(result.predictions.dass21_probs[index] * 100).toFixed(1)}%
                  </span>
                </div>
              ))}
            </div>
            <div className="text-xs text-gray-500">
              Total score: {result.metadata.dass21_values.reduce((a, b) => a + b, 0)}/21
            </div>
          </CardContent>
        </Card>

        {/* Voice Results */}
        <Card className="border-0 shadow-lg bg-white/90 backdrop-blur-sm">
          <CardHeader className="pb-3">
            <CardTitle className="text-lg flex items-center gap-2">
              <Mic className="w-5 h-5 text-teal-600" />
              Voice Analysis
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            {result.predictions.voice_probs ? (
              <div className="space-y-2">
                {["Low", "Medium", "High"].map((level, index) => (
                  <div key={level} className="flex justify-between items-center">
                    <span className="text-sm text-gray-600">{level}</span>
                    <span className="text-sm font-semibold">
                      {(result.predictions.voice_probs[index] * 100).toFixed(1)}%
                    </span>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center text-gray-500 text-sm">
                <Info className="w-4 h-4 mx-auto mb-1" />
                Default distribution used
              </div>
            )}
            <div className="text-xs text-gray-500">
              {result.metadata.voice_provided ? "Voice data provided" : "Voice data not provided"}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Explanations Section */}
      {result.explanations && (
        <Card className="border-0 shadow-lg bg-white/90 backdrop-blur-sm">
          <CardHeader>
            <CardTitle className="text-xl flex items-center gap-2">
              <Lightbulb className="w-6 h-6 text-yellow-600" />
              AI Explanations
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-6">
            {/* Fusion Explanation */}
            {result.explanations.fusion.available && (
              <div className="space-y-3">
                <h4 className="font-semibold text-lg">Fusion Analysis</h4>
                <p className="text-gray-700 bg-gray-50 p-3 rounded-lg">
                  {result.explanations.fusion.summary}
                </p>
                
                {/* Modality Contributions */}
                {/* <div className="space-y-2">
                  <h5 className="font-medium">Modality Contributions:</h5>
                  <div className="grid md:grid-cols-3 gap-3">
                    {result.explanations.fusion.modality_contributions.map((modality, index) => (
                      <div key={index} className="p-3 bg-gray-50 rounded-lg">
                        <div className="font-medium text-sm capitalize">{modality.modality}</div>
                        <div className="text-xs text-gray-600">
                          Confidence: {(modality.confidence * 100).toFixed(1)}%
                        </div>
                        <div className="text-xs text-gray-600">
                          Contribution: {(modality.contribution_score * 100).toFixed(1)}%
                        </div>
                      </div>
                    ))}
                  </div>
                </div> */}
              </div>
            )}

            {/* Physiological Explanations */}
            {result.explanations.physiological.available && result.explanations.physiological.feature_importance.length > 0 && (
              <div className="space-y-3">
                <h4 className="font-semibold text-lg">Key Physiological Factors</h4>
                <p className="text-gray-700 bg-gray-50 p-3 rounded-lg">
                  {result.explanations.physiological.summary}
                </p>
                <div className="space-y-2">
                  {result.explanations.physiological.feature_importance.slice(0, 5).map((feature, index) => (
                    <div key={index} className="flex justify-between items-center p-2 bg-gray-50 rounded">
                      <span className="text-sm text-gray-700">
                        {formatFeatureName(feature.feature)}
                      </span>
                      <div className="flex items-center gap-2">
                        <span className="text-xs text-gray-500">
                          {feature.importance > 0 ? "Increases stress" : "Decreases stress"}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Questionnaire Explanations */}
            {result.explanations.questionnaire.available && result.explanations.questionnaire.feature_importance.length > 0 && (
              <div className="space-y-3">
                <h4 className="font-semibold text-lg">Questionnaire Insights</h4>
                <p className="text-gray-700 bg-gray-50 p-3 rounded-lg">
                  {result.explanations.questionnaire.summary}
                </p>
                <div className="space-y-2">
                  {result.explanations.questionnaire.feature_importance.slice(0, 3).map((feature, index) => (
                    <div key={index} className="flex justify-between items-center p-2 bg-gray-50 rounded">
                      <span className="text-sm text-gray-700">
                        {formatFeatureName(feature.feature)}
                      </span>
                      <div className="flex items-center gap-2">
                        <span className="text-xs text-gray-500">Score: {feature.value}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Voice Explanations */}
            {result.explanations.voice.available && result.explanations.voice.feature_importance.length > 0 && (
              <div className="space-y-3">
                <h4 className="font-semibold text-lg">Voice Analysis Insights</h4>
                <p className="text-gray-700 bg-gray-50 p-3 rounded-lg">
                  {result.explanations.voice.summary}
                </p>
                <div className="space-y-2">
                  {result.explanations.voice.feature_importance.slice(0, 3).map((feature, index) => (
                    <div key={index} className="flex justify-between items-center p-2 bg-gray-50 rounded">
                      <span className="text-sm text-gray-700">
                        {formatFeatureName(feature.feature)}
                      </span>
                      <div className="flex items-center gap-2">
                        <span className="text-xs text-gray-500">
                          {(feature.value * 100).toFixed(1)}% probability
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* Metadata */}
      <Card className="border-0 shadow-lg bg-white/90 backdrop-blur-sm">
        <CardHeader>
          <CardTitle className="text-lg flex items-center gap-2">
            <Info className="w-5 h-5 text-blue-600" />
            Analysis Details
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid md:grid-cols-2 gap-4 text-sm">
            <div className="space-y-2">
              <div className="flex justify-between">
                <span className="text-gray-600">Physiological Windows:</span>
                <span className="font-medium">{result.metadata.physio_windows}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Features Extracted:</span>
                <span className="font-medium">{result.metadata.physio_features}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Modalities Used:</span>
                <span className="font-medium">{result.metadata.modalities_used.filter(Boolean).join(", ")}</span>
              </div>
            </div>
            <div className="space-y-2">
              <div className="flex justify-between">
                <span className="text-gray-600">DASS-21 Total Score:</span>
                <span className="font-medium">{result.metadata.dass21_values.reduce((a, b) => a + b, 0)}/21</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Voice Data:</span>
                <span className="font-medium">{result.metadata.voice_provided ? "Provided" : "Not provided"}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Analysis Time:</span>
                <span className="font-medium">{new Date().toLocaleTimeString()}</span>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

export default Results 