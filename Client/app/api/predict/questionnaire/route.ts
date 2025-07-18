import { type NextRequest, NextResponse } from "next/server"

export async function POST(request: NextRequest) {
  try {
    const { responses } = await request.json()

    // Simulate DASS-21 analysis
    // In a real implementation, this would use your ML model
    const totalScore = responses.reduce((sum: number, score: number) => sum + score, 0)
    const maxScore = 63 // 21 questions * 3 max score
    const percentage = Math.round((totalScore / maxScore) * 100)

    let level = "Low"
    let recommendations = [
      "Continue maintaining healthy lifestyle habits",
      "Practice regular relaxation techniques",
      "Maintain social connections",
    ]

    if (percentage > 60) {
      level = "High"
      recommendations = [
        "Consider speaking with a mental health professional",
        "Practice stress reduction techniques daily",
        "Ensure adequate sleep and exercise",
        "Consider mindfulness or meditation practices",
      ]
    } else if (percentage > 30) {
      level = "Moderate"
      recommendations = [
        "Monitor stress levels regularly",
        "Implement stress management strategies",
        "Maintain work-life balance",
        "Consider relaxation techniques",
      ]
    }

    return NextResponse.json({
      score: percentage,
      level,
      recommendations,
      method: "questionnaire",
      timestamp: new Date().toISOString(),
    })
  } catch (error) {
    return NextResponse.json({ error: "Failed to analyze questionnaire data" }, { status: 500 })
  }
}
