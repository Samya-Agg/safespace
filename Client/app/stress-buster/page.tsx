"use client"

import { useState } from "react"
import Link from "next/link"
import { ArrowLeft, Gamepad2, Brain, Zap, Star } from "lucide-react"
import DinosaurGame from "../components/DinosaurGame"
import MemoryGame from "../components/MemoryGame"

export default function StressBusterPage() {
  const [selectedGame, setSelectedGame] = useState<string | null>(null)

  if (selectedGame === "dinosaur") {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 via-indigo-50 to-teal-50 pt-20">
        <div className="container mx-auto px-4 py-8">
          <button
            onClick={() => setSelectedGame(null)}
            className="flex items-center space-x-2 text-blue-600 hover:text-blue-800 transition-colors mb-6 group"
          >
            <ArrowLeft size={20} className="group-hover:-translate-x-1 transition-transform" />
            <span>Back to Games</span>
          </button>
          
          <div className="text-center mb-8">
            <h1 className="text-4xl md:text-5xl font-bold bg-gradient-to-r from-green-600 to-emerald-600 bg-clip-text text-transparent mb-4">
              🦕 Dinosaur Jump
            </h1>
            <p className="text-gray-600 max-w-2xl mx-auto">
              Help the dinosaur jump over obstacles and beat your high score!
            </p>
          </div>
          
          <DinosaurGame />
        </div>
      </div>
    )
  }

  if (selectedGame === "memory") {
    return (
      <div className="min-h-screen bg-gradient-to-br from-purple-50 via-pink-50 to-indigo-50 pt-20">
        <div className="container mx-auto px-4 py-8">
          <button
            onClick={() => setSelectedGame(null)}
            className="flex items-center space-x-2 text-purple-600 hover:text-purple-800 transition-colors mb-6 group"
          >
            <ArrowLeft size={20} className="group-hover:-translate-x-1 transition-transform" />
            <span>Back to Games</span>
          </button>
          
          <div className="text-center mb-8">
            <h1 className="text-4xl md:text-5xl font-bold bg-gradient-to-r from-purple-600 to-pink-600 bg-clip-text text-transparent mb-4">
              🧠 Memory Challenge
            </h1>
            <p className="text-gray-600 max-w-2xl mx-auto">
              Test your memory skills and improve cognitive function!
            </p>
          </div>
          
          <MemoryGame />
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-indigo-50 to-teal-50 pt-20 relative overflow-hidden">
      {/* Animated background elements */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-20 left-10 w-20 h-20 bg-blue-200 rounded-full opacity-20 animate-pulse"></div>
        <div className="absolute top-40 right-20 w-16 h-16 bg-teal-200 rounded-full opacity-30 animate-bounce"></div>
        <div className="absolute bottom-40 left-20 w-12 h-12 bg-indigo-200 rounded-full opacity-25 animate-pulse"></div>
        <div className="absolute bottom-20 right-10 w-24 h-24 bg-cyan-200 rounded-full opacity-20 animate-bounce"></div>
        <div className="absolute top-1/2 left-1/2 w-32 h-32 bg-purple-200 rounded-full opacity-10 animate-spin" style={{ animationDuration: '20s' }}></div>
      </div>

      <div className="container mx-auto px-4 py-6 relative z-10 flex flex-col justify-center min-h-[calc(100vh-5rem)]">
        {/* Header Section */}
        <div className="text-center mb-8">
          <div className="mb-4">
            <h1 className="text-4xl md:text-5xl font-bold bg-gradient-to-r from-blue-600 via-purple-600 to-teal-600 bg-clip-text text-transparent mb-4">
              🎮 StressBuster
            </h1>
            <div className="w-40 h-1 bg-gradient-to-r from-blue-500 via-purple-500 to-teal-500 mx-auto rounded-full animate-pulse"></div>
          </div>
          <p className="text-lg md:text-xl text-gray-700 max-w-3xl mx-auto leading-relaxed mb-6">
            Take a mental break with our collection of stress-relieving games. 
            Perfect for clearing your mind and boosting your mood!
          </p>
          <div className="flex justify-center mb-6">
            <Link
              href="/check"
              className="inline-flex items-center space-x-2 px-6 py-3 bg-gradient-to-r from-blue-500 to-teal-500 text-white text-sm font-medium rounded-full hover:from-blue-600 hover:to-teal-600 transition-all duration-200 shadow-lg hover:shadow-xl"
            >
              <span>🔍</span>
              <span>Ready to check your stress level?</span>
            </Link>
          </div>
          <div className="flex justify-center space-x-6 text-sm text-gray-600">
            <span className="flex items-center bg-white/50 px-4 py-2 rounded-full backdrop-blur-sm">
              <Zap className="w-4 h-4 text-yellow-500 mr-2" />
              Instant Stress Relief
            </span>
            <span className="flex items-center bg-white/50 px-4 py-2 rounded-full backdrop-blur-sm">
              <Brain className="w-4 h-4 text-purple-500 mr-2" />
              Cognitive Boost
            </span>
            <span className="flex items-center bg-white/50 px-4 py-2 rounded-full backdrop-blur-sm">
              <Star className="w-4 h-4 text-blue-500 mr-2" />
              Fun & Engaging
            </span>
          </div>
        </div>
        
        {/* Game Cards */}
        <div className="max-w-6xl mx-auto">
          <div className="grid md:grid-cols-2 gap-6">
            {/* Dinosaur Game Card */}
            <div 
              className="group cursor-pointer transform hover:scale-105 transition-all duration-300"
              onClick={() => setSelectedGame("dinosaur")}
            >
              <div className="bg-white/80 backdrop-blur-sm rounded-2xl shadow-xl border border-white/20 overflow-hidden hover:shadow-2xl transition-all duration-300">
                <div className="bg-gradient-to-br from-green-400 to-emerald-600 p-8 text-white relative overflow-hidden">
                  <div className="absolute top-0 right-0 w-32 h-32 bg-white/10 rounded-full -translate-y-16 translate-x-16"></div>
                  <div className="absolute bottom-0 left-0 w-24 h-24 bg-white/10 rounded-full translate-y-12 -translate-x-12"></div>
                  
                  <div className="relative z-10">
                    <div className="text-6xl mb-4 group-hover:animate-bounce">🦕</div>
                    <h3 className="text-2xl font-bold mb-2">Dinosaur Jump</h3>
                    <p className="text-green-100 text-sm">Classic endless runner game</p>
                  </div>
                </div>
                
                <div className="p-6">
                  <p className="text-gray-600 mb-4 leading-relaxed">
                    Help our friendly dinosaur jump over obstacles in this classic endless runner. 
                    Perfect for quick stress relief and improving reaction time.
                  </p>
                  
                  <div className="flex items-center justify-between mb-4">
                    <div className="flex space-x-4 text-xs text-gray-500">
                      <span className="flex items-center">
                        <div className="w-2 h-2 bg-green-500 rounded-full mr-1"></div>
                        Reflexes
                      </span>
                      <span className="flex items-center">
                        <div className="w-2 h-2 bg-blue-500 rounded-full mr-1"></div>
                        Focus
                      </span>
                      <span className="flex items-center">
                        <div className="w-2 h-2 bg-yellow-500 rounded-full mr-1"></div>
                        Fun
                      </span>
                    </div>
                    <div className="text-xs text-gray-400">2-5 min</div>
                  </div>
                  
                  <div className="flex justify-end">
                    <div className="bg-gradient-to-r from-green-500 to-emerald-600 text-white px-4 py-2 rounded-lg text-sm font-medium group-hover:from-green-600 group-hover:to-emerald-700 transition-all duration-200 flex items-center space-x-2">
                      <Gamepad2 size={16} />
                      <span>Play Now</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Memory Game Card */}
            <div 
              className="group cursor-pointer transform hover:scale-105 transition-all duration-300"
              onClick={() => setSelectedGame("memory")}
            >
              <div className="bg-white/80 backdrop-blur-sm rounded-2xl shadow-xl border border-white/20 overflow-hidden hover:shadow-2xl transition-all duration-300">
                <div className="bg-gradient-to-br from-purple-400 to-pink-600 p-8 text-white relative overflow-hidden">
                  <div className="absolute top-0 right-0 w-32 h-32 bg-white/10 rounded-full -translate-y-16 translate-x-16"></div>
                  <div className="absolute bottom-0 left-0 w-24 h-24 bg-white/10 rounded-full translate-y-12 -translate-x-12"></div>
                  
                  <div className="relative z-10">
                    <div className="text-6xl mb-4 group-hover:animate-pulse">🧠</div>
                    <h3 className="text-2xl font-bold mb-2">Memory Challenge</h3>
                    <p className="text-purple-100 text-sm">Card matching brain trainer</p>
                  </div>
                </div>
                
                <div className="p-6">
                  <p className="text-gray-600 mb-4 leading-relaxed">
                    Exercise your brain with this colorful memory matching game. 
                    Improve cognitive function while having fun and reducing stress.
                  </p>
                  
                  <div className="flex items-center justify-between mb-4">
                    <div className="flex space-x-4 text-xs text-gray-500">
                      <span className="flex items-center">
                        <div className="w-2 h-2 bg-purple-500 rounded-full mr-1"></div>
                        Memory
                      </span>
                      <span className="flex items-center">
                        <div className="w-2 h-2 bg-pink-500 rounded-full mr-1"></div>
                        Logic
                      </span>
                      <span className="flex items-center">
                        <div className="w-2 h-2 bg-indigo-500 rounded-full mr-1"></div>
                        Calm
                      </span>
                    </div>
                    <div className="text-xs text-gray-400">3-7 min</div>
                  </div>
                  
                  <div className="flex justify-end">
                    <div className="bg-gradient-to-r from-purple-500 to-pink-600 text-white px-4 py-2 rounded-lg text-sm font-medium group-hover:from-purple-600 group-hover:to-pink-700 transition-all duration-200 flex items-center space-x-2">
                      <Brain size={16} />
                      <span>Play Now</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
