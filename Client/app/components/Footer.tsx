"use client"

import { Github, Heart } from "lucide-react"

export default function Footer() {
  return (
    <footer className="bg-gray-900 text-white py-12 sm:py-14 md:py-16">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center">
          <a
            href="https://github.com/arshchatrath/SafeSpace"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center space-x-2 sm:space-x-3 text-blue-400 hover:text-blue-300 mb-4 sm:mb-6 text-lg sm:text-xl font-semibold transition-all duration-300 hover:scale-105"
          >
            <Github size={24} className="sm:w-7 sm:h-7" />
            <span>GitHub Repository</span>
          </a>
          <p className="flex items-center justify-center space-x-2 sm:space-x-3 text-gray-400 text-base sm:text-lg">
            <span>Made with</span>
            <Heart className="w-5 h-5 sm:w-6 sm:h-6 text-red-500 animate-pulse" fill="currentColor" />
            <span>by SafeSpace AI Team</span>
          </p>
        </div>
      </div>
    </footer>
  )
}
