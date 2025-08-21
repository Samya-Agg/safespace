"use client"

import { useState, useEffect, useRef, useCallback } from "react"

interface GameState {
  isRunning: boolean
  isGameOver: boolean
  score: number
  highScore: number
}

interface DinoState {
  y: number
  velocity: number
  isJumping: boolean
}

interface Obstacle {
  x: number
  width: number
  height: number
}

export default function DinosaurGame() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const animationRef = useRef<number>()
  
  const [gameState, setGameState] = useState<GameState>({
    isRunning: false,
    isGameOver: false,
    score: 0,
    highScore: 0
  })

  // Game constants
  const CANVAS_WIDTH = 800
  const CANVAS_HEIGHT = 200
  const GROUND_HEIGHT = 20
  const DINO_WIDTH = 20
  const DINO_HEIGHT = 20
  const GRAVITY = 0.6
  const JUMP_FORCE = -12
  const GAME_SPEED = 3
  const OBSTACLE_WIDTH = 15
  const OBSTACLE_HEIGHT = 30

  // Game state refs
  const dinoRef = useRef<DinoState>({
    y: CANVAS_HEIGHT - GROUND_HEIGHT - DINO_HEIGHT,
    velocity: 0,
    isJumping: false
  })
  
  const obstaclesRef = useRef<Obstacle[]>([])
  const gameSpeedRef = useRef(GAME_SPEED)
  const scoreRef = useRef(0)

  // Initialize high score from localStorage
  useEffect(() => {
    const savedHighScore = localStorage.getItem('stressbuster-highscore')
    if (savedHighScore) {
      setGameState(prev => ({ ...prev, highScore: parseInt(savedHighScore) }))
    }
  }, [])

  const jump = useCallback(() => {
    if (!dinoRef.current.isJumping && gameState.isRunning) {
      dinoRef.current.velocity = JUMP_FORCE
      dinoRef.current.isJumping = true
    }
  }, [gameState.isRunning])

  const startGame = useCallback(() => {
    dinoRef.current = {
      y: CANVAS_HEIGHT - GROUND_HEIGHT - DINO_HEIGHT,
      velocity: 0,
      isJumping: false
    }
    obstaclesRef.current = []
    gameSpeedRef.current = GAME_SPEED
    scoreRef.current = 0
    
    setGameState(prev => ({
      ...prev,
      isRunning: true,
      isGameOver: false,
      score: 0
    }))
  }, [])

  const gameOver = useCallback(() => {
    setGameState(prev => {
      const newHighScore = Math.max(prev.highScore, scoreRef.current)
      localStorage.setItem('stressbuster-highscore', newHighScore.toString())
      
      return {
        ...prev,
        isRunning: false,
        isGameOver: true,
        score: scoreRef.current,
        highScore: newHighScore
      }
    })
  }, [])

  const checkCollision = useCallback((dino: DinoState, obstacle: Obstacle): boolean => {
    const dinoScale = 1.5
    const scaledWidth = DINO_WIDTH * dinoScale
    const scaledHeight = DINO_HEIGHT * dinoScale
    
    const dinoLeft = 50 + 5 // Add small margin for better gameplay
    const dinoRight = dinoLeft + scaledWidth - 10 // Reduce hitbox slightly
    const dinoTop = dino.y + 5
    const dinoBottom = dino.y + scaledHeight - 5

    const obstacleLeft = obstacle.x + 2
    const obstacleRight = obstacle.x + obstacle.width - 2
    const obstacleTop = CANVAS_HEIGHT - GROUND_HEIGHT - obstacle.height
    const obstacleBottom = CANVAS_HEIGHT - GROUND_HEIGHT

    return (
      dinoRight > obstacleLeft &&
      dinoLeft < obstacleRight &&
      dinoBottom > obstacleTop &&
      dinoTop < obstacleBottom
    )
  }, [])

  const gameLoop = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas || !gameState.isRunning) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // Create gradient background (sky)
    const skyGradient = ctx.createLinearGradient(0, 0, 0, CANVAS_HEIGHT)
    skyGradient.addColorStop(0, '#87CEEB')
    skyGradient.addColorStop(0.7, '#E0F6FF')
    skyGradient.addColorStop(1, '#F0F9FF')
    ctx.fillStyle = skyGradient
    ctx.fillRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT - GROUND_HEIGHT)

    // Draw animated clouds
    const time = Date.now() * 0.001
    ctx.fillStyle = 'rgba(255, 255, 255, 0.8)'
    for (let i = 0; i < 3; i++) {
      const cloudX = (200 + i * 250 + time * 10) % (CANVAS_WIDTH + 100)
      const cloudY = 30 + i * 15
      // Cloud shape
      ctx.beginPath()
      ctx.arc(cloudX, cloudY, 15, 0, Math.PI * 2)
      ctx.arc(cloudX + 15, cloudY, 20, 0, Math.PI * 2)
      ctx.arc(cloudX + 30, cloudY, 15, 0, Math.PI * 2)
      ctx.arc(cloudX + 15, cloudY - 10, 15, 0, Math.PI * 2)
      ctx.fill()
    }

    // Draw ground with texture
    const groundGradient = ctx.createLinearGradient(0, CANVAS_HEIGHT - GROUND_HEIGHT, 0, CANVAS_HEIGHT)
    groundGradient.addColorStop(0, '#22c55e')
    groundGradient.addColorStop(0.3, '#16a34a')
    groundGradient.addColorStop(1, '#15803d')
    ctx.fillStyle = groundGradient
    ctx.fillRect(0, CANVAS_HEIGHT - GROUND_HEIGHT, CANVAS_WIDTH, GROUND_HEIGHT)

    // Add ground texture lines
    ctx.strokeStyle = '#166534'
    ctx.lineWidth = 1
    for (let i = 0; i < CANVAS_WIDTH; i += 20) {
      ctx.beginPath()
      ctx.moveTo(i, CANVAS_HEIGHT - GROUND_HEIGHT + 5)
      ctx.lineTo(i + 10, CANVAS_HEIGHT - GROUND_HEIGHT + 5)
      ctx.stroke()
    }

    // Update dino physics
    const dino = dinoRef.current
    dino.velocity += GRAVITY
    dino.y += dino.velocity

    // Ground collision
    const groundY = CANVAS_HEIGHT - GROUND_HEIGHT - DINO_HEIGHT
    if (dino.y >= groundY) {
      dino.y = groundY
      dino.velocity = 0
      dino.isJumping = false
    }

    // Draw enhanced and appealing dino with detailed animation
    const dinoX = 50
    const dinoScale = 1.5 // Make dino bigger
    const scaledWidth = DINO_WIDTH * dinoScale
    const scaledHeight = DINO_HEIGHT * dinoScale
    
    ctx.save()
    
    // Dino shadow for depth
    ctx.fillStyle = 'rgba(0, 0, 0, 0.2)'
    ctx.beginPath()
    ctx.ellipse(dinoX + scaledWidth/2, CANVAS_HEIGHT - GROUND_HEIGHT + 2, scaledWidth/2, 4, 0, 0, Math.PI * 2)
    ctx.fill()
    
    // Main dino body with gradient
    const dinoGradient = ctx.createLinearGradient(dinoX, dino.y, dinoX + scaledWidth, dino.y + scaledHeight)
    dinoGradient.addColorStop(0, '#10b981')
    dinoGradient.addColorStop(0.5, '#059669')
    dinoGradient.addColorStop(1, '#047857')
    ctx.fillStyle = dinoGradient
    ctx.beginPath()
    ctx.roundRect(dinoX, dino.y, scaledWidth, scaledHeight, 6)
    ctx.fill()
    
    // Dino belly (lighter color)
    ctx.fillStyle = '#34d399'
    ctx.beginPath()
    ctx.roundRect(dinoX + 4, dino.y + 8, scaledWidth - 8, scaledHeight - 12, 4)
    ctx.fill()
    
    // Dino head (larger and more detailed)
    const headGradient = ctx.createLinearGradient(dinoX + scaledWidth - 5, dino.y - 8, dinoX + scaledWidth + 8, dino.y + 5)
    headGradient.addColorStop(0, '#10b981')
    headGradient.addColorStop(1, '#047857')
    ctx.fillStyle = headGradient
    ctx.beginPath()
    ctx.roundRect(dinoX + scaledWidth - 5, dino.y - 8, 15, 12, 6)
    ctx.fill()
    
    // Dino snout
    ctx.fillStyle = '#059669'
    ctx.beginPath()
    ctx.roundRect(dinoX + scaledWidth + 8, dino.y - 3, 6, 6, 3)
    ctx.fill()
    
    // Dino nostrils
    ctx.fillStyle = '#047857'
    ctx.beginPath()
    ctx.arc(dinoX + scaledWidth + 10, dino.y - 1, 0.5, 0, Math.PI * 2)
    ctx.fill()
    ctx.beginPath()
    ctx.arc(dinoX + scaledWidth + 12, dino.y - 1, 0.5, 0, Math.PI * 2)
    ctx.fill()
    
    // Dino eyes (more expressive)
    ctx.fillStyle = '#ffffff'
    ctx.beginPath()
    ctx.arc(dinoX + scaledWidth - 2, dino.y - 3, 2.5, 0, Math.PI * 2)
    ctx.fill()
    ctx.beginPath()
    ctx.arc(dinoX + scaledWidth + 3, dino.y - 3, 2.5, 0, Math.PI * 2)
    ctx.fill()
    
    // Eye pupils (animated blinking)
    const blinkTime = Math.sin(time * 0.5) > 0.9 ? 0 : 1
    if (blinkTime) {
      ctx.fillStyle = '#000000'
      ctx.beginPath()
      ctx.arc(dinoX + scaledWidth - 1, dino.y - 3, 1.5, 0, Math.PI * 2)
      ctx.fill()
      ctx.beginPath()
      ctx.arc(dinoX + scaledWidth + 4, dino.y - 3, 1.5, 0, Math.PI * 2)
      ctx.fill()
      
      // Eye shine
      ctx.fillStyle = '#ffffff'
      ctx.beginPath()
      ctx.arc(dinoX + scaledWidth - 0.5, dino.y - 3.5, 0.5, 0, Math.PI * 2)
      ctx.fill()
      ctx.beginPath()
      ctx.arc(dinoX + scaledWidth + 4.5, dino.y - 3.5, 0.5, 0, Math.PI * 2)
      ctx.fill()
    }
    
    // Dino spikes on back
    ctx.fillStyle = '#047857'
    for (let i = 0; i < 3; i++) {
      const spikeX = dinoX + 8 + i * 6
      const spikeY = dino.y - 2
      ctx.beginPath()
      ctx.moveTo(spikeX, spikeY)
      ctx.lineTo(spikeX + 3, spikeY - 4)
      ctx.lineTo(spikeX + 6, spikeY)
      ctx.closePath()
      ctx.fill()
    }
    
    // Animated legs with more detail
    const legOffset = dino.isJumping ? 0 : Math.sin(time * 12) * 3
    const legOffset2 = dino.isJumping ? 0 : Math.sin(time * 12 + Math.PI) * 3
    
    // Back legs
    ctx.fillStyle = '#059669'
    ctx.beginPath()
    ctx.roundRect(dinoX + 4, dino.y + scaledHeight, 4, 8 + legOffset, 2)
    ctx.fill()
    ctx.beginPath()
    ctx.roundRect(dinoX + 10, dino.y + scaledHeight, 4, 8 + legOffset2, 2)
    ctx.fill()
    
    // Front legs
    ctx.beginPath()
    ctx.roundRect(dinoX + 16, dino.y + scaledHeight, 4, 6 + legOffset2, 2)
    ctx.fill()
    ctx.beginPath()
    ctx.roundRect(dinoX + 22, dino.y + scaledHeight, 4, 6 + legOffset, 2)
    ctx.fill()
    
    // Dino feet
    ctx.fillStyle = '#047857'
    ctx.beginPath()
    ctx.roundRect(dinoX + 3, dino.y + scaledHeight + 8 + legOffset, 6, 3, 1)
    ctx.fill()
    ctx.beginPath()
    ctx.roundRect(dinoX + 9, dino.y + scaledHeight + 8 + legOffset2, 6, 3, 1)
    ctx.fill()
    ctx.beginPath()
    ctx.roundRect(dinoX + 15, dino.y + scaledHeight + 6 + legOffset2, 6, 3, 1)
    ctx.fill()
    ctx.beginPath()
    ctx.roundRect(dinoX + 21, dino.y + scaledHeight + 6 + legOffset, 6, 3, 1)
    ctx.fill()
    
    // Dino tail (more detailed and animated)
    const tailWag = Math.sin(time * 8) * 2
    ctx.fillStyle = '#047857'
    ctx.beginPath()
    ctx.roundRect(dinoX - 8, dino.y + 12 + tailWag, 12, 6, 3)
    ctx.fill()
    ctx.beginPath()
    ctx.roundRect(dinoX - 12, dino.y + 14 + tailWag, 8, 4, 2)
    ctx.fill()
    
    ctx.restore()

    // Update obstacles
    obstaclesRef.current = obstaclesRef.current.filter(obstacle => {
      obstacle.x -= gameSpeedRef.current
      return obstacle.x + obstacle.width > 0
    })

    // Add new obstacles with varied types
    if (obstaclesRef.current.length === 0 || 
        obstaclesRef.current[obstaclesRef.current.length - 1].x < CANVAS_WIDTH - 200) {
      const obstacleType = Math.random()
      obstaclesRef.current.push({
        x: CANVAS_WIDTH,
        width: obstacleType > 0.7 ? OBSTACLE_WIDTH * 2 : OBSTACLE_WIDTH,
        height: OBSTACLE_HEIGHT + Math.random() * 20
      })
    }

    // Draw enhanced obstacles (cacti)
    obstaclesRef.current.forEach(obstacle => {
      const obstacleY = CANVAS_HEIGHT - GROUND_HEIGHT - obstacle.height
      
      // Cactus body
      const cactusGradient = ctx.createLinearGradient(obstacle.x, obstacleY, obstacle.x + obstacle.width, obstacleY)
      cactusGradient.addColorStop(0, '#16a34a')
      cactusGradient.addColorStop(0.5, '#22c55e')
      cactusGradient.addColorStop(1, '#15803d')
      ctx.fillStyle = cactusGradient
      ctx.beginPath()
      ctx.roundRect(obstacle.x, obstacleY, obstacle.width, obstacle.height, 3)
      ctx.fill()
      
      // Cactus arms
      if (obstacle.width > OBSTACLE_WIDTH) {
        ctx.fillStyle = '#16a34a'
        ctx.beginPath()
        ctx.roundRect(obstacle.x - 5, obstacleY + 10, 8, 15, 2)
        ctx.fill()
        ctx.beginPath()
        ctx.roundRect(obstacle.x + obstacle.width - 3, obstacleY + 8, 8, 12, 2)
        ctx.fill()
      }
      
      // Cactus spikes
      ctx.strokeStyle = '#15803d'
      ctx.lineWidth = 1
      for (let i = 0; i < obstacle.height; i += 8) {
        ctx.beginPath()
        ctx.moveTo(obstacle.x, obstacleY + i)
        ctx.lineTo(obstacle.x - 2, obstacleY + i + 2)
        ctx.stroke()
        ctx.beginPath()
        ctx.moveTo(obstacle.x + obstacle.width, obstacleY + i)
        ctx.lineTo(obstacle.x + obstacle.width + 2, obstacleY + i + 2)
        ctx.stroke()
      }
    })

    // Check collisions (update collision detection for scaled dino)
    for (const obstacle of obstaclesRef.current) {
      if (checkCollision(dino, obstacle)) {
        gameOver()
        return
      }
    }

    // Update score and speed
    scoreRef.current += 1
    gameSpeedRef.current = GAME_SPEED + Math.floor(scoreRef.current / 1000) * 0.5

    // Update React state for real-time score display
    setGameState(prev => ({
      ...prev,
      score: scoreRef.current
    }))

    // Draw enhanced score with shadow
    ctx.save()
    ctx.shadowColor = 'rgba(0, 0, 0, 0.3)'
    ctx.shadowBlur = 2
    ctx.shadowOffsetX = 1
    ctx.shadowOffsetY = 1
    ctx.fillStyle = '#1f2937'
    ctx.font = 'bold 18px Arial'
    ctx.fillText(`Score: ${Math.floor(scoreRef.current / 10)}`, CANVAS_WIDTH - 140, 35)
    ctx.restore()

    // Draw speed indicator
    ctx.fillStyle = '#6b7280'
    ctx.font = '12px Arial'
    ctx.fillText(`Speed: ${gameSpeedRef.current.toFixed(1)}x`, CANVAS_WIDTH - 140, 55)

    animationRef.current = requestAnimationFrame(gameLoop)
  }, [gameState.isRunning, checkCollision, gameOver])

  useEffect(() => {
    if (gameState.isRunning) {
      animationRef.current = requestAnimationFrame(gameLoop)
    } else if (animationRef.current) {
      cancelAnimationFrame(animationRef.current)
    }

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
    }
  }, [gameState.isRunning, gameLoop])

  useEffect(() => {
    const handleKeyPress = (e: KeyboardEvent) => {
      if (e.code === 'Space') {
        e.preventDefault()
        if (gameState.isGameOver || !gameState.isRunning) {
          startGame()
        } else {
          jump()
        }
      }
    }

    const handleClick = () => {
      if (gameState.isGameOver || !gameState.isRunning) {
        startGame()
      } else {
        jump()
      }
    }

    window.addEventListener('keydown', handleKeyPress)
    const canvas = canvasRef.current
    if (canvas) {
      canvas.addEventListener('click', handleClick)
    }

    return () => {
      window.removeEventListener('keydown', handleKeyPress)
      if (canvas) {
        canvas.removeEventListener('click', handleClick)
      }
    }
  }, [gameState.isGameOver, gameState.isRunning, jump, startGame])

  return (
    <div className="flex flex-col items-center space-y-4">
      <div className="bg-white rounded-lg shadow-lg p-4">
        <canvas
          ref={canvasRef}
          width={CANVAS_WIDTH}
          height={CANVAS_HEIGHT}
          className="border-2 border-gray-300 rounded cursor-pointer"
          style={{ maxWidth: '100%', height: 'auto' }}
        />
      </div>
      
      <div className="flex items-center space-x-8 text-center">
        <div>
          <p className="text-sm text-gray-500">Score</p>
          <p className="text-2xl font-bold text-blue-600">{Math.floor(gameState.score / 10)}</p>
        </div>
        <div>
          <p className="text-sm text-gray-500">High Score</p>
          <p className="text-2xl font-bold text-teal-600">{Math.floor(gameState.highScore / 10)}</p>
        </div>
      </div>

      {!gameState.isRunning && (
        <div className="text-center">
          <button
            onClick={startGame}
            className="px-6 py-3 bg-gradient-to-r from-blue-600 to-teal-600 text-white font-semibold rounded-lg hover:from-blue-700 hover:to-teal-700 transition-all duration-200 shadow-lg"
          >
            {gameState.isGameOver ? 'Play Again' : 'Start Game'}
          </button>
          {gameState.isGameOver && (
            <p className="mt-2 text-red-600 font-semibold">Game Over!</p>
          )}
        </div>
      )}
    </div>
  )
}
