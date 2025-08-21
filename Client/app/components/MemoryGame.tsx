"use client"

import { useState, useEffect, useCallback } from "react"
import { RotateCcw, Trophy, Clock, Star } from "lucide-react"

interface Card {
  id: number
  emoji: string
  isFlipped: boolean
  isMatched: boolean
}

interface GameStats {
  moves: number
  matches: number
  timeElapsed: number
  isComplete: boolean
}

const CARD_EMOJIS = [
  "🌟", "🎨", "🎭", "🎪", "🎯", "🎲", "🎸", "🎺",
  "🌈", "🦋", "🌸", "🍀", "🌺", "🎀", "💎", "🔮"
]

export default function MemoryGame() {
  const [cards, setCards] = useState<Card[]>([])
  const [flippedCards, setFlippedCards] = useState<number[]>([])
  const [gameStats, setGameStats] = useState<GameStats>({
    moves: 0,
    matches: 0,
    timeElapsed: 0,
    isComplete: false
  })
  const [gameStarted, setGameStarted] = useState(false)
  const [difficulty, setDifficulty] = useState<'easy' | 'medium' | 'hard'>('medium')

  const getDifficultySettings = (level: 'easy' | 'medium' | 'hard') => {
    switch (level) {
      case 'easy': return { pairs: 6, gridCols: 'grid-cols-3' }
      case 'medium': return { pairs: 8, gridCols: 'grid-cols-4' }
      case 'hard': return { pairs: 12, gridCols: 'grid-cols-4' }
    }
  }

  const initializeGame = useCallback(() => {
    const { pairs } = getDifficultySettings(difficulty)
    const selectedEmojis = CARD_EMOJIS.slice(0, pairs)
    const gameCards = [...selectedEmojis, ...selectedEmojis]
      .sort(() => Math.random() - 0.5)
      .map((emoji, index) => ({
        id: index,
        emoji,
        isFlipped: false,
        isMatched: false
      }))

    setCards(gameCards)
    setFlippedCards([])
    setGameStats({
      moves: 0,
      matches: 0,
      timeElapsed: 0,
      isComplete: false
    })
    setGameStarted(true)
  }, [difficulty])

  const handleCardClick = useCallback((cardId: number) => {
    if (flippedCards.length === 2 || flippedCards.includes(cardId)) return
    
    const card = cards.find(c => c.id === cardId)
    if (card?.isMatched) return

    setFlippedCards(prev => [...prev, cardId])
    setCards(prev => prev.map(card => 
      card.id === cardId ? { ...card, isFlipped: true } : card
    ))
  }, [cards, flippedCards])

  useEffect(() => {
    if (flippedCards.length === 2) {
      const [first, second] = flippedCards
      const firstCard = cards.find(c => c.id === first)
      const secondCard = cards.find(c => c.id === second)

      setGameStats(prev => ({ ...prev, moves: prev.moves + 1 }))

      setTimeout(() => {
        if (firstCard?.emoji === secondCard?.emoji) {
          // Match found
          setCards(prev => prev.map(card => 
            card.id === first || card.id === second 
              ? { ...card, isMatched: true }
              : card
          ))
          setGameStats(prev => ({ ...prev, matches: prev.matches + 1 }))
        } else {
          // No match - flip back
          setCards(prev => prev.map(card => 
            card.id === first || card.id === second 
              ? { ...card, isFlipped: false }
              : card
          ))
        }
        setFlippedCards([])
      }, 1000)
    }
  }, [flippedCards, cards])

  useEffect(() => {
    const { pairs } = getDifficultySettings(difficulty)
    if (gameStats.matches === pairs && gameStarted) {
      setGameStats(prev => ({ ...prev, isComplete: true }))
    }
  }, [gameStats.matches, difficulty, gameStarted])

  useEffect(() => {
    let interval: NodeJS.Timeout
    if (gameStarted && !gameStats.isComplete) {
      interval = setInterval(() => {
        setGameStats(prev => ({ ...prev, timeElapsed: prev.timeElapsed + 1 }))
      }, 1000)
    }
    return () => clearInterval(interval)
  }, [gameStarted, gameStats.isComplete])

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60)
    const secs = seconds % 60
    return `${mins}:${secs.toString().padStart(2, '0')}`
  }

  const getPerformanceRating = () => {
    const { pairs } = getDifficultySettings(difficulty)
    const efficiency = pairs / gameStats.moves
    if (efficiency > 0.8) return { stars: 5, text: "Perfect!" }
    if (efficiency > 0.6) return { stars: 4, text: "Excellent!" }
    if (efficiency > 0.4) return { stars: 3, text: "Good!" }
    if (efficiency > 0.3) return { stars: 2, text: "Not bad!" }
    return { stars: 1, text: "Keep trying!" }
  }

  if (!gameStarted) {
    return (
      <div className="max-w-2xl mx-auto">
        <div className="bg-white/80 backdrop-blur-sm rounded-2xl shadow-xl border border-white/20 p-8">
          <div className="text-center mb-8">
            <div className="text-6xl mb-4">🧠</div>
            <h2 className="text-3xl font-bold bg-gradient-to-r from-purple-600 to-pink-600 bg-clip-text text-transparent mb-4">
              Memory Challenge
            </h2>
            <p className="text-gray-600 leading-relaxed">
              Test your memory by matching pairs of cards. Choose your difficulty level and start training your brain!
            </p>
          </div>

          <div className="mb-8">
            <h3 className="text-lg font-semibold text-gray-800 mb-4 text-center">Choose Difficulty</h3>
            <div className="grid grid-cols-3 gap-4">
              {(['easy', 'medium', 'hard'] as const).map((level) => {
                const { pairs } = getDifficultySettings(level)
                return (
                  <button
                    key={level}
                    onClick={() => setDifficulty(level)}
                    className={`p-4 rounded-xl border-2 transition-all duration-200 ${
                      difficulty === level
                        ? 'border-purple-500 bg-purple-50 text-purple-700'
                        : 'border-gray-200 bg-white hover:border-purple-300'
                    }`}
                  >
                    <div className="text-2xl mb-2">
                      {level === 'easy' ? '😊' : level === 'medium' ? '🤔' : '🤯'}
                    </div>
                    <div className="font-semibold capitalize">{level}</div>
                    <div className="text-sm text-gray-500">{pairs} pairs</div>
                  </button>
                )
              })}
            </div>
          </div>

          <button
            onClick={initializeGame}
            className="w-full bg-gradient-to-r from-purple-600 to-pink-600 text-white font-semibold py-4 px-6 rounded-xl hover:from-purple-700 hover:to-pink-700 transition-all duration-200 shadow-lg flex items-center justify-center space-x-2"
          >
            <Star size={20} />
            <span>Start Game</span>
          </button>
        </div>
      </div>
    )
  }

  const { gridCols } = getDifficultySettings(difficulty)

  return (
    <div className="max-w-4xl mx-auto">
      {/* Game Stats */}
      <div className="flex justify-center space-x-6 mb-8">
        <div className="bg-white/80 backdrop-blur-sm rounded-lg px-4 py-2 shadow-lg border border-white/20">
          <div className="flex items-center space-x-2">
            <Clock size={16} className="text-blue-600" />
            <span className="font-semibold">{formatTime(gameStats.timeElapsed)}</span>
          </div>
        </div>
        <div className="bg-white/80 backdrop-blur-sm rounded-lg px-4 py-2 shadow-lg border border-white/20">
          <div className="flex items-center space-x-2">
            <Trophy size={16} className="text-yellow-600" />
            <span className="font-semibold">{gameStats.moves} moves</span>
          </div>
        </div>
        <div className="bg-white/80 backdrop-blur-sm rounded-lg px-4 py-2 shadow-lg border border-white/20">
          <div className="flex items-center space-x-2">
            <Star size={16} className="text-purple-600" />
            <span className="font-semibold">{gameStats.matches} matches</span>
          </div>
        </div>
      </div>

      {/* Game Board */}
      <div className="bg-white/80 backdrop-blur-sm rounded-2xl shadow-xl border border-white/20 p-6 mb-6">
        <div className={`grid ${gridCols} gap-4 max-w-2xl mx-auto`}>
          {cards.map((card) => (
            <div
              key={card.id}
              onClick={() => handleCardClick(card.id)}
              className={`aspect-square rounded-xl cursor-pointer transition-all duration-300 transform hover:scale-105 ${
                card.isFlipped || card.isMatched
                  ? 'bg-gradient-to-br from-purple-400 to-pink-500 text-white shadow-lg'
                  : 'bg-gradient-to-br from-gray-200 to-gray-300 hover:from-gray-300 hover:to-gray-400 shadow-md'
              } ${card.isMatched ? 'ring-4 ring-green-400 ring-opacity-50' : ''}`}
            >
              <div className="w-full h-full flex items-center justify-center text-3xl font-bold">
                {card.isFlipped || card.isMatched ? (
                  <span className="animate-in zoom-in-50 duration-300">{card.emoji}</span>
                ) : (
                  <span className="text-gray-400">?</span>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Controls */}
      <div className="flex justify-center space-x-4">
        <button
          onClick={initializeGame}
          className="bg-gradient-to-r from-blue-600 to-teal-600 text-white font-semibold py-3 px-6 rounded-lg hover:from-blue-700 hover:to-teal-700 transition-all duration-200 shadow-lg flex items-center space-x-2"
        >
          <RotateCcw size={18} />
          <span>New Game</span>
        </button>
        <button
          onClick={() => setGameStarted(false)}
          className="bg-gradient-to-r from-purple-600 to-pink-600 text-white font-semibold py-3 px-6 rounded-lg hover:from-purple-700 hover:to-pink-700 transition-all duration-200 shadow-lg"
        >
          Change Difficulty
        </button>
      </div>

      {/* Victory Modal */}
      {gameStats.isComplete && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="bg-white rounded-2xl shadow-2xl max-w-md w-full mx-4 transform animate-in fade-in-0 zoom-in-95 duration-300">
            <div className="bg-gradient-to-r from-purple-600 to-pink-600 rounded-t-2xl p-6 text-white text-center">
              <div className="text-6xl mb-4">🎉</div>
              <h3 className="text-2xl font-bold mb-2">Congratulations!</h3>
              <p className="text-purple-100">You completed the memory challenge!</p>
            </div>
            
            <div className="p-6 text-center">
              <div className="mb-6">
                <div className="flex justify-center space-x-1 mb-2">
                  {[...Array(5)].map((_, i) => (
                    <Star
                      key={i}
                      size={24}
                      className={`${
                        i < getPerformanceRating().stars
                          ? 'text-yellow-400 fill-current'
                          : 'text-gray-300'
                      }`}
                    />
                  ))}
                </div>
                <p className="text-lg font-semibold text-gray-800">
                  {getPerformanceRating().text}
                </p>
              </div>

              <div className="grid grid-cols-2 gap-4 mb-6 text-sm">
                <div className="bg-gray-50 rounded-lg p-3">
                  <div className="font-semibold text-gray-800">Time</div>
                  <div className="text-blue-600">{formatTime(gameStats.timeElapsed)}</div>
                </div>
                <div className="bg-gray-50 rounded-lg p-3">
                  <div className="font-semibold text-gray-800">Moves</div>
                  <div className="text-purple-600">{gameStats.moves}</div>
                </div>
              </div>

              <div className="flex space-x-3">
                <button
                  onClick={initializeGame}
                  className="flex-1 bg-gradient-to-r from-purple-600 to-pink-600 text-white font-semibold py-3 px-4 rounded-lg hover:from-purple-700 hover:to-pink-700 transition-all duration-200"
                >
                  Play Again
                </button>
                <button
                  onClick={() => setGameStarted(false)}
                  className="px-4 py-3 text-gray-500 hover:text-gray-700 transition-colors font-medium"
                >
                  New Level
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
