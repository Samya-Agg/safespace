"use client"

import { useEffect, useRef, useState } from "react"
import { gsap } from "gsap"
import Link from "next/link"
import { Button } from "@/components/ui/button"
import { Brain, Zap, Shield, Activity, Heart, Eye } from "lucide-react"

interface HeroSectionProps {
  onLearnMore: () => void
}

export default function HeroSection({ onLearnMore }: HeroSectionProps) {
  const heroRef = useRef<HTMLDivElement>(null)
  const floatingElementsRef = useRef<HTMLDivElement>(null)
  const particlesRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const ctx = gsap.context(() => {
      // Create timeline for orchestrated animations
      const tl = gsap.timeline()

      // Hero text animations with stagger
      tl.fromTo(
        ".hero-title-line",
        { 
          y: 120, 
          opacity: 0, 
          scale: 0.8,
          rotationX: 90,
          transformOrigin: "center bottom"
        },
        { 
          y: 0, 
          opacity: 1, 
          scale: 1,
          rotationX: 0,
          duration: 1.4, 
          ease: "power4.out",
          stagger: 0.15
        }
      )
      .fromTo(
        ".hero-subtitle",
        { 
          y: 60, 
          opacity: 0,
          filter: "blur(10px)"
        },
        { 
          y: 0, 
          opacity: 1,
          filter: "blur(0px)",
          duration: 1.2, 
          ease: "power3.out" 
        },
        "-=0.8"
      )
      .fromTo(
        ".hero-buttons",
        { 
          y: 50, 
          opacity: 0,
          scale: 0.9
        },
        { 
          y: 0, 
          opacity: 1,
          scale: 1,
          duration: 1, 
          ease: "power3.out" 
        },
        "-=0.6"
      )

      // Enhanced floating icons animation
      tl.fromTo(
        ".floating-icon",
        { 
          scale: 0, 
          rotation: -270, 
          opacity: 0,
          y: 100
        },
        {
          scale: 1,
          rotation: 0,
          opacity: 1,
          y: 0,
          duration: 1.8,
          stagger: 0.15,
          ease: "elastic.out(1, 0.6)",
        },
        "-=1.2"
      )

      // Particle system animation
      gsap.fromTo(
        ".particle",
        { 
          scale: 0, 
          opacity: 0,
          y: 50
        },
        {
          scale: 1,
          opacity: 0.6,
          y: 0,
          duration: 2,
          stagger: 0.05,
          ease: "power2.out",
          delay: 1.5
        }
      )

      // Continuous animations
      gsap.to(".floating-icon", {
        y: -25,
        rotation: 5,
        duration: 3,
        repeat: -1,
        yoyo: true,
        ease: "power2.inOut",
        stagger: 0.4,
      })

      // Enhanced background orbs with morphing
      gsap.to(".bg-orb-1", {
        rotation: 360,
        scale: 1.1,
        duration: 25,
        repeat: -1,
        ease: "none",
      })

      gsap.to(".bg-orb-2", {
        rotation: -360,
        scale: 0.9,
        duration: 30,
        repeat: -1,
        ease: "none",
      })

      gsap.to(".bg-orb-3", {
        rotation: 180,
        scale: 1.05,
        duration: 35,
        repeat: -1,
        yoyo: true,
        ease: "power1.inOut",
      })

      // Floating particles animation
      gsap.to(".particle", {
        y: -100,
        opacity: 0,
        duration: 8,
        repeat: -1,
        ease: "none",
        stagger: {
          each: 0.2,
          repeat: -1
        }
      })

      // // Scroll indicator pulse
      // gsap.to(".scroll-indicator", {
      //   scale: 1.1,
      //   opacity: 0.7,
      //   duration: 2,
      //   repeat: -1,
      //   yoyo: true,
      //   ease: "power2.inOut"
      // })

      // Text shimmer effect
      gsap.to(".text-shimmer", {
        backgroundPosition: "200% center",
        duration: 3,
        repeat: -1,
        ease: "none"
      })
    })

    return () => ctx.revert()
  }, [])

  // Generate particles
  const particles = Array.from({ length: 50 }, (_, i) => i)
  const floatingIcons = [
    { icon: Brain, color: "from-indigo-400 to-purple-500", size: "w-16 h-16", delay: 0 },
    { icon: Zap, color: "from-yellow-400 to-orange-500", size: "w-14 h-14", delay: 0.2 },
    { icon: Shield, color: "from-emerald-400 to-teal-500", size: "w-15 h-15", delay: 0.4 },
    { icon: Activity, color: "from-pink-400 to-red-500", size: "w-12 h-12", delay: 0.6 },
    { icon: Heart, color: "from-rose-400 to-pink-500", size: "w-13 h-13", delay: 0.8 },
    { icon: Eye, color: "from-cyan-400 to-blue-500", size: "w-14 h-14", delay: 1.0 },
  ]

  return (
    <section
      ref={heroRef}
      className="relative min-h-screen flex items-center justify-center overflow-hidden"
      style={{
        background: `radial-gradient(circle at 50% 50%, rgba(99, 102, 241, 0.15) 0%, transparent 50%), 
                     linear-gradient(135deg, #0f172a 0%, #1e293b 25%, #0f172a 50%, #1e293b 75%, #0f172a 100%)`
      }}
    >
      {/* Enhanced Animated Background */}
      <div className="absolute inset-0">
        {/* Multiple layered orbs with different blend modes */}
        <div className="bg-orb-1 absolute -top-48 -left-48 w-96 h-96 md:w-[600px] md:h-[600px] bg-gradient-to-r from-blue-500/20 to-indigo-500/20 rounded-full blur-3xl mix-blend-multiply"></div>
        <div className="bg-orb-2 absolute -bottom-48 -right-48 w-[400px] h-[400px] md:w-[700px] md:h-[700px] bg-gradient-to-r from-teal-500/20 to-cyan-500/20 rounded-full blur-3xl mix-blend-multiply"></div>
        <div className="bg-orb-3 absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-[500px] h-[500px] md:w-[800px] md:h-[800px] bg-gradient-to-r from-purple-500/10 to-pink-500/10 rounded-full blur-3xl mix-blend-multiply"></div>
        
        {/* Additional accent orbs */}
        <div className="bg-orb-1 absolute top-20 right-20 w-32 h-32 md:w-48 md:h-48 bg-gradient-to-r from-yellow-400/15 to-orange-400/15 rounded-full blur-2xl"></div>
        <div className="bg-orb-2 absolute bottom-32 left-32 w-40 h-40 md:w-60 md:h-60 bg-gradient-to-r from-emerald-400/15 to-teal-400/15 rounded-full blur-2xl"></div>

        {/* Animated grid pattern */}
        <div 
          className="absolute inset-0 opacity-30"
          style={{
            backgroundImage: `radial-gradient(circle at 1px 1px, rgba(255,255,255,0.15) 1px, transparent 0)`,
            backgroundSize: '50px 50px',
            animation: 'gridMove 20s linear infinite'
          }}
        ></div>
      </div>

      {/* Floating Particles */}
      <div ref={particlesRef} className="absolute inset-0 pointer-events-none">
        {particles.map((i) => (
          <div
            key={i}
            className="particle absolute w-1 h-1 bg-white rounded-full opacity-60"
            style={{
              left: `${Math.random() * 100}%`,
              top: `${100 + Math.random() * 100}%`,
              animationDelay: `${Math.random() * 8}s`,
            }}
          />
        ))}
      </div>

      {/* Enhanced Floating Icons */}
      <div ref={floatingElementsRef} className="absolute inset-0 pointer-events-none">
        {floatingIcons.map((item, index) => {
          const positions = [
            { top: "15%", left: "8%", transform: "rotate(15deg)" },
            { top: "25%", right: "12%", transform: "rotate(-10deg)" },
            { bottom: "30%", left: "10%", transform: "rotate(25deg)" },
            { top: "45%", right: "8%", transform: "rotate(-15deg)" },
            { bottom: "20%", right: "20%", transform: "rotate(10deg)" },
            { top: "60%", left: "15%", transform: "rotate(-20deg)" },
          ]
          
          const position = positions[index]
          const IconComponent = item.icon

          return (
            <div
              key={index}
              className="floating-icon absolute"
              style={position}
            >
              <div className={`${item.size} bg-gradient-to-br ${item.color} rounded-2xl flex items-center justify-center shadow-2xl backdrop-blur-sm border border-white/20 hover:scale-110 transition-transform duration-300`}>
                <IconComponent className="w-1/2 h-1/2 text-white drop-shadow-lg" />
              </div>
            </div>
          )
        })}
      </div>

      {/* Main Content */}
      <div className="relative z-10 text-center px-4 sm:px-6 max-w-7xl mx-auto">
        <div className="space-y-6 md:space-y-10">
          <h1 className="text-4xl sm:text-5xl md:text-6xl lg:text-7xl xl:text-8xl font-bold leading-tight">
            <span className="hero-title-line block text-shimmer bg-gradient-to-r from-white via-blue-100 to-cyan-100 bg-clip-text text-transparent bg-size-200 animate-shimmer">
              Your Mental Health,
            </span>
            <span className="hero-title-line block bg-gradient-to-r from-blue-400 via-teal-400 to-cyan-400 bg-clip-text text-transparent mt-2 drop-shadow-sm">
              Empowered by AI
            </span>
          </h1>

          <p className="hero-subtitle text-lg sm:text-xl md:text-2xl lg:text-3xl text-blue-100/90 max-w-5xl mx-auto font-light leading-relaxed px-2 backdrop-blur-sm">
            Detect stress through voice, sensors, and psychology with cutting-edge multimodal AI technology.
          </p>

          <div className="hero-buttons flex flex-col sm:flex-row gap-4 sm:gap-6 justify-center items-center pt-6 md:pt-10">
            <Link href="/check" className="w-full sm:w-auto group">
              <Button
                size="lg"
                className="relative w-full sm:w-auto bg-gradient-to-r from-blue-500 via-indigo-500 to-teal-500 hover:from-blue-600 hover:via-indigo-600 hover:to-teal-600 text-white px-10 sm:px-14 py-4 sm:py-5 text-lg sm:text-xl font-semibold rounded-full shadow-2xl hover:shadow-blue-500/30 transition-all duration-500 hover:scale-105 border-0 overflow-hidden group-hover:shadow-2xl"
              >
                <span className="relative z-10">Check Yours Now</span>
                <div className="absolute inset-0 bg-gradient-to-r from-white/20 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500 rounded-full"></div>
              </Button>
            </Link>
            <Button
              size="lg"
              variant="outline"
              onClick={onLearnMore}
              className="group relative w-full sm:w-auto border-2 border-white/30 text-white hover:bg-white/10 hover:border-white/60 px-10 sm:px-14 py-4 sm:py-5 text-lg sm:text-xl font-semibold rounded-full bg-white/5 backdrop-blur-md hover:scale-105 transition-all duration-500 overflow-hidden hover:shadow-xl"
            >
              <span className="relative z-10">Learn More</span>
              <div className="absolute inset-0 bg-gradient-to-r from-white/10 via-white/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500 rounded-full"></div>
            </Button>
          </div>
        </div>

        {/* ...removed scroll indicator... */}
      </div>

      <style jsx>{`
        @keyframes gridMove {
          0% { transform: translate(0, 0); }
          100% { transform: translate(50px, 50px); }
        }
        
        @keyframes shimmer {
          0% { background-position: -200% center; }
          100% { background-position: 200% center; }
        }
        
        .bg-size-200 {
          background-size: 200% 100%;
        }
        
        .animate-shimmer {
          animation: shimmer 3s ease-in-out infinite;
        }
      `}</style>
    </section>
  )
}
