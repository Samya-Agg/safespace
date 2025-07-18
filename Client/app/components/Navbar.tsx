"use client"

import Link from "next/link"
import { useState, useEffect } from "react"
import { Menu, X, Github } from "lucide-react"

export default function Navbar() {
  const [isOpen, setIsOpen] = useState(false)
  const [scrolled, setScrolled] = useState(false)

  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 50)
    }
    window.addEventListener("scroll", handleScroll)
    return () => window.removeEventListener("scroll", handleScroll)
  }, [])

  return (
    <nav
      className={`fixed top-0 w-full z-50 transition-all duration-300 ${
        scrolled
          ? "bg-white/95 backdrop-blur-md border-b border-gray-200 shadow-lg"
          : "bg-gradient-to-r from-blue-700 via-blue-600 to-teal-600 bg-opacity-95 backdrop-blur-md border-b border-blue-800 shadow-lg"
      }`}
      style={{
        backgroundColor: scrolled ? undefined : "rgba(29, 78, 216, 0.95)",
        color: scrolled ? undefined : "#fff",
        borderBottom: scrolled ? undefined : "2px solid #2563eb"
      }}
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-14 sm:h-16">
          <Link
            href="/"
            className={`text-xl sm:text-2xl font-bold transition-colors duration-300 ${
              scrolled
                ? "bg-gradient-to-r from-blue-600 to-teal-600 bg-clip-text text-transparent"
                : "text-white drop-shadow-lg"
            }`}
          >
            SafeSpace AI
          </Link>

          {/* Desktop Navigation */}
          <div className="hidden md:flex items-center space-x-6 lg:space-x-8">
            <Link
              href="/"
              className={`transition-colors duration-300 hover:scale-105 ${
                scrolled ? "text-gray-700 hover:text-blue-600" : "text-white/90 hover:text-white" 
              } font-semibold`}
              style={!scrolled ? { textShadow: "0 2px 8px rgba(0,0,0,0.25)" } : {}}
            >
              Home
            </Link>
            <a
              href="/#contributors"
              className={`transition-colors duration-300 hover:scale-105 ${
                scrolled ? "text-gray-700 hover:text-blue-600" : "text-white/90 hover:text-white" 
              } font-semibold`}
              style={!scrolled ? { textShadow: "0 2px 8px rgba(0,0,0,0.25)" } : {}}
            >
              Contributors
            </a>
            <a
              href="https://github.com/arshchatrath/SafeSpace"
              target="_blank"
              rel="noopener noreferrer"
              className={`flex items-center space-x-2 transition-all duration-300 hover:scale-105 ${
                scrolled ? "text-gray-700 hover:text-blue-600" : "text-white/90 hover:text-white" 
              } font-semibold`}
              style={!scrolled ? { textShadow: "0 2px 8px rgba(0,0,0,0.25)" } : {}}
            >
              <Github size={18} />
              <span className="hidden lg:inline">GitHub</span>
            </a>
          </div>

          {/* Mobile menu button */}
          <div className="md:hidden">
            <button
              onClick={() => setIsOpen(!isOpen)}
              className={`transition-colors duration-300 ${
                scrolled ? "text-gray-700 hover:text-blue-600" : "text-white hover:text-blue-200"
              }`}
            >
              {isOpen ? <X size={24} /> : <Menu size={24} />}
            </button>
          </div>
        </div>

        {/* Mobile Navigation */}
        {isOpen && (
          <div className="md:hidden">
            <div className="px-2 pt-2 pb-3 space-y-1 bg-gradient-to-r from-blue-700 via-blue-600 to-teal-600 bg-opacity-95 backdrop-blur-md border-t border-blue-800 rounded-b-lg shadow-lg">
              <Link
                href="/"
                className="block px-3 py-2 text-white font-semibold hover:text-blue-200 hover:bg-blue-800/60 rounded-lg transition-all duration-200"
                style={{ textShadow: "0 2px 8px rgba(0,0,0,0.25)" }}
                onClick={() => setIsOpen(false)}
              >
                Home
              </Link>
              <a
                href="/#contributors"
                className="block px-3 py-2 text-white font-semibold hover:text-blue-200 hover:bg-blue-800/60 rounded-lg transition-all duration-200"
                style={{ textShadow: "0 2px 8px rgba(0,0,0,0.25)" }}
                onClick={() => setIsOpen(false)}
              >
                Contributors
              </a>
              <a
                href="https://github.com/safespace-ai"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center space-x-2 px-3 py-2 text-white font-semibold hover:text-blue-200 hover:bg-blue-800/60 rounded-lg transition-all duration-200"
                style={{ textShadow: "0 2px 8px rgba(0,0,0,0.25)" }}
                onClick={() => setIsOpen(false)}
              >
                <Github size={18} />
                <span>GitHub</span>
              </a>
            </div>
          </div>
        )}
      </div>
    </nav>
  )
}
