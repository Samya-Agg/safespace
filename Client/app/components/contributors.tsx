"use client"

import { useEffect } from "react"
import { gsap } from "gsap"
import { ScrollTrigger } from "gsap/ScrollTrigger"
import { Github, Linkedin, Mail, MapPin } from "lucide-react"
import { Card, CardContent } from "@/components/ui/card"

gsap.registerPlugin(ScrollTrigger)

const contributors = [
  {
    name: "Dr. Rajendra Kumar Roul",
    gradient: "from-blue-600 to-indigo-600",
    initials: "RK",
    img: "sir.jpg",
    expertise: ["Machine Learning", "DSA", "Computer Vision"]
  },
  {
    name: "Arsh Chatrath",
    github: "arshchatrath",
    linkedin: "arshchatrath",
    gradient: "from-blue-600 to-indigo-600",
    initials: "AC",
    img: "Arsh.jpg",
    expertise: ["Machine Learning", "NLP", "Computer Vision"]
  },
  {
    name: "Kabir Oberoi",
    github: "kabiroberoi",
    linkedin: "kabir-oberoi-2a8480299",
    gradient: "from-teal-600 to-cyan-600",
    initials: "KO",
    img: "Kabir.png",
    expertise: ["Edge Computing", "IoT", "Distributed Systems"]
  },
  {
    name: "Devansh",
    github: "devanshkumar",
    linkedin: "devansh-kumar",
    gradient: "from-purple-600 to-violet-600",
    initials: "DK",
    img: "Devansh.png",
    expertise: ["DSP", "Audio Processing", "Algorithm Design"]
  },
  {
    name: "Durvish",
    github: "durvishkhurana",
    linkedin: "durvishkhurana",
    gradient: "from-emerald-600 to-teal-600",
    initials: "DK",
    img: "durvish.png",
    expertise: ["Voice Analysis", "Audio ML", "Speech Recognition"]
  },
  {
    name: "Lakshita",
    github: "lakshitasharma",
    linkedin: "lakshita-sharma",
    gradient: "from-orange-600 to-red-600",
    initials: "LS",
    img: "lakshita.jpg",
    expertise: ["User Experience", "Interface Design", "Design Systems"]
  },
  {
    name: "Khushpreet",
    github: "khushpreet",
    linkedin: "khushpreet",
    gradient: "from-indigo-600 to-purple-600",
    initials: "K",
    img: "khushpreet.jpg",
    expertise: ["React", "Node.js", "System Architecture"]
  },
  {
    name: "Samya Aggarwal",
    github: "Samya-Agg",
    linkedin: "samya-aggarwal-5729182a6",
    gradient: "from-cyan-600 to-blue-600",
    initials: "SA",
    img: "samya.png",
    expertise: ["Django", "FastAPI", "Research"]
  },
];


export default function ContributorsSection() {
  useEffect(() => {
    const ctx = gsap.context(() => {
      // Animate contributor cards
      gsap.fromTo(
        ".contributor-card",
        { 
          scale: 0.95, 
          opacity: 0, 
          y: 40,
          rotateX: 15
        },
        {
          scale: 1,
          opacity: 1,
          y: 0,
          rotateX: 0,
          duration: 0.6,
          stagger: 0.08,
          ease: "power2.out",
          scrollTrigger: {
            trigger: ".contributors-grid",
            start: "top 80%",
            toggleActions: "play none none reverse"
          },
        },
      )

      // Animate section title
      gsap.fromTo(
        ".section-title",
        { y: 20, opacity: 0 },
        {
          y: 0,
          opacity: 1,
          duration: 0.8,
          ease: "power2.out",
          scrollTrigger: {
            trigger: ".section-title",
            start: "top 90%",
          },
        },
      )
    })

    return () => ctx.revert()
  }, [])

  return (
    <section id="contributors" className="py-16 md:py-24 lg:py-32 bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50 relative overflow-hidden">
      {/* Background decorative elements */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-20 left-10 w-96 h-96 bg-blue-200/20 rounded-full blur-3xl"></div>
        <div className="absolute top-40 right-20 w-80 h-80 bg-indigo-200/20 rounded-full blur-3xl"></div>
        <div className="absolute bottom-20 left-1/4 w-[500px] h-[500px] bg-purple-200/20 rounded-full blur-3xl"></div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
        {/* Section Header */}
        <div className="text-center mb-16 md:mb-20 section-title">
          <h2 className="text-4xl md:text-5xl lg:text-6xl font-bold mb-6 text-slate-900">
            Meet the Minds Behind
            <span className="block bg-gradient-to-r from-blue-600 via-indigo-600 to-purple-600 bg-clip-text text-transparent">
              SafeSpace AI
            </span>
          </h2>
          <div className="w-24 h-1 bg-gradient-to-r from-blue-600 to-indigo-600 mx-auto mb-6"></div>
        </div>

        {/* Contributors Grid */}
        <div className="contributors-grid space-y-8 mb-16">
          {/* First Row - 4 Cards */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6 md:gap-8">
            {contributors.slice(0, 4).map((contributor, index) => (
              <ContributorCard key={index} contributor={contributor} />
            ))}
          </div>

          {/* Second Row - 4 Cards */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6 md:gap-8">
            {contributors.slice(4, 8).map((contributor, index) => (
              <ContributorCard key={index + 4} contributor={contributor} />
            ))}
          </div>
        </div>
      </div>
    </section>
  )
}

function ContributorCard({ contributor }) {
  return (
    <Card className="contributor-card group border-0 bg-white/80 backdrop-blur-sm hover:bg-white transition-all duration-300 hover:shadow-xl hover:-translate-y-1 rounded-2xl overflow-hidden shadow-sm">
      <CardContent className="p-0 relative">
        {/* Profile Image/Avatar */}
        <div className="relative w-full aspect-[4/3] overflow-hidden">
          {contributor.img ? (
            <img
              src={contributor.img}
              alt={`${contributor.name} - ${contributor.role}`}
              className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
            />
          ) : (
            <div className={`w-full h-full flex items-center justify-center bg-gradient-to-br ${contributor.gradient} relative`}>
              <span className="text-3xl font-bold text-white">
                {contributor.initials}
              </span>
              <div className="absolute inset-0 bg-black/0 group-hover:bg-black/5 transition-colors duration-300"></div>
            </div>
          )}
          
          {/* Location Badge removed */}
        </div>

        {/* Content */}
        <div className="p-6">
          <div className="mb-4">
            <h3 className="text-xl font-bold text-slate-900 mb-1">
              {contributor.name}
            </h3>
            <p className="text-blue-600 font-semibold text-sm mb-1">
              {contributor.role}
            </p>
          </div>

          {/* Expertise Tags */}
          <div className="mb-4">
            <div className="flex flex-wrap gap-1">
              {contributor.expertise.slice(0, 2).map((skill, index) => (
                <span 
                  key={index}
                  className="text-xs bg-slate-100 text-slate-700 px-2 py-1 rounded-md"
                >
                  {skill}
                </span>
              ))}
              {contributor.expertise.length > 2 && (
                <span className="text-xs bg-slate-100 text-slate-700 px-2 py-1 rounded-md">
                  +{contributor.expertise.length - 2}
                </span>
              )}
            </div>
          </div>

          {/* Social Links */}
          <div className="flex justify-center gap-3">
            <a
              href={`https://linkedin.com/in/${contributor.linkedin}`}
              target="_blank"
              rel="noopener noreferrer"
              className="p-2 bg-blue-100 hover:bg-blue-200 rounded-lg transition-colors duration-200 group/link"
              title="LinkedIn Profile"
              aria-label={`${contributor.name}'s LinkedIn profile`}
            >
              <Linkedin className="w-4 h-4 text-blue-600 group-hover/link:text-blue-700" />
            </a>
            <a
              href={`https://github.com/${contributor.github}`}
              target="_blank"
              rel="noopener noreferrer"
              className="p-2 bg-slate-100 hover:bg-slate-200 rounded-lg transition-colors duration-200 group/link"
              title="GitHub Profile"
              aria-label={`${contributor.name}'s GitHub profile`}
            >
              <Github className="w-4 h-4 text-slate-700 group-hover/link:text-slate-900" />
            </a>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
