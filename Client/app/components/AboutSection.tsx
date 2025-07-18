"use client"

import { useEffect, useState } from "react"
import { Mic, Activity, FileText, Brain, Cpu, CheckCircle, Target, Shield, Zap, Users, Award, TrendingUp } from "lucide-react"

// Simulated GSAP animations with CSS transitions and Intersection Observer
const useScrollAnimation = () => {
  const [visibleElements, setVisibleElements] = useState(new Set())

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            setVisibleElements(prev => new Set([...prev, entry.target.dataset.animate]))
          }
        })
      },
      { threshold: 0.1, rootMargin: '0px 0px -50px 0px' }
    )

    const elements = document.querySelectorAll('[data-animate]')
    elements.forEach(el => observer.observe(el))

    return () => observer.disconnect()
  }, [])

  return visibleElements
}

const AnimatedCard = ({ children, className = "", delay = 0, ...props }) => {
  const visibleElements = useScrollAnimation()
  const isVisible = visibleElements.has(props['data-animate'])
  
  return (
    <div
      className={`transition-all duration-700 ease-out ${
        isVisible 
          ? 'opacity-100 translate-y-0 scale-100' 
          : 'opacity-0 translate-y-8 scale-95'
      } ${className}`}
      style={{ transitionDelay: `${delay}ms` }}
      {...props}
    >
      {children}
    </div>
  )
}

const StatCard = ({ value, label, icon: Icon, color, delay = 0 }) => (
  <AnimatedCard 
    data-animate={`stat-${value}`} 
    delay={delay}
    className="group"
  >
    <div className="relative bg-white/95 backdrop-blur p-3 sm:p-4 rounded-xl shadow-md hover:shadow-lg transition-all duration-400 hover:-translate-y-1 border border-white/30 overflow-hidden min-w-[120px] max-w-[170px] mx-auto flex flex-col items-center">
      <div className="absolute inset-0 bg-gradient-to-br from-white/60 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-400"></div>
      <div className="relative z-10 flex flex-col items-center">
        <div className={`w-8 h-8 ${color} rounded-lg flex items-center justify-center mb-2 group-hover:scale-105 transition-transform duration-300`}>
          <Icon className="w-4 h-4 text-white" />
        </div>
        <div className="text-2xl sm:text-3xl font-extrabold mb-1 bg-gradient-to-r from-gray-800 to-gray-600 bg-clip-text text-transparent text-center">
          {value}
        </div>
        <div className="text-xs text-gray-600 font-medium text-center leading-tight">{label}</div>
      </div>
    </div>
  </AnimatedCard>
)

const FeatureCard = ({ icon: Icon, title, description, badge, gradient, delay = 0 }) => (
  <AnimatedCard 
    data-animate={`feature-${title}`} 
    delay={delay}
    className="group h-full"
  >
    <div className={`relative h-full bg-gradient-to-br ${gradient} rounded-2xl p-8 shadow-lg hover:shadow-2xl transition-all duration-500 hover:-translate-y-3 border border-white/20 overflow-hidden`}>
      <div className="absolute inset-0 bg-gradient-to-br from-white/20 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500"></div>
      <div className="relative z-10 h-full flex flex-col">
        <div className="w-20 h-20 bg-white/90 backdrop-blur-sm rounded-2xl flex items-center justify-center mx-auto mb-6 shadow-lg group-hover:scale-110 group-hover:rotate-3 transition-all duration-300">
          <Icon className="w-10 h-10 text-gray-700" />
        </div>
        <h4 className="text-2xl font-bold mb-4 text-gray-900 text-center">
          {title}
        </h4>
        <p className="text-gray-700 mb-6 leading-relaxed flex-grow text-center">
          {description}
        </p>
        <div className="bg-white/80 backdrop-blur-sm px-4 py-2 rounded-full text-sm font-semibold text-gray-700 text-center">
          {badge}
        </div>
      </div>
    </div>
  </AnimatedCard>
)

const TechSpec = ({ icon: Icon, text, delay = 0 }) => (
  <AnimatedCard 
    data-animate={`spec-${text}`} 
    delay={delay}
    className="group"
  >
    <div className="flex items-center space-x-3 p-4 bg-white/70 backdrop-blur-sm rounded-xl hover:bg-white/90 transition-all duration-300 border border-white/30">
      <div className="w-8 h-8 bg-gradient-to-r from-green-500 to-emerald-500 rounded-lg flex items-center justify-center flex-shrink-0 group-hover:scale-110 transition-transform duration-300">
        <Icon className="w-4 h-4 text-white" />
      </div>
      <span className="text-gray-700 font-medium">{text}</span>
    </div>
  </AnimatedCard>
)

export default function AboutSection() {
  const visibleElements = useScrollAnimation()

return (
    <section className="relative pt-20 lg:pt-32 pb-4 lg:pb-6 overflow-hidden">
      {/* Background with animated gradient */}
      <div className="absolute inset-0 bg-gradient-to-br from-slate-50 via-blue-50/30 to-purple-50/30">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_25%_25%,rgba(59,130,246,0.1),transparent_50%)]"></div>
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_75%_75%,rgba(147,51,234,0.1),transparent_50%)]"></div>
      </div>
      
      <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Enhanced Header */}
        <div className="text-center mb-20">
          <AnimatedCard data-animate="main-title">
            <h2 className="text-5xl sm:text-6xl lg:text-7xl font-bold mb-8 bg-gradient-to-r from-gray-900 via-blue-800 to-purple-800 bg-clip-text text-transparent leading-tight">
              What is SafeSpace AI?
            </h2>
            <div className="w-32 h-1.5 bg-gradient-to-r from-blue-600 via-teal-500 to-purple-600 mx-auto rounded-full mb-8"></div>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto leading-relaxed">
              The future of stress detection through intelligent multimodal analysis
            </p>
          </AnimatedCard>
        </div>

        {/* Hero Section with Enhanced Layout */}
        <div className="mb-24">
          <div className="grid lg:grid-cols-2 gap-16 items-center mb-20">
            <AnimatedCard data-animate="hero-text" className="space-y-8">
              <div className="space-y-6">
                <p className="text-xl text-gray-700 leading-relaxed">
                  SafeSpace AI is a <span className="font-bold text-blue-600">revolutionary real-time stress detection system</span> that combines accuracy, interpretability, and deployability. Recognizing that stress is complex and varies between individuals and contexts, our platform takes a comprehensive multimodal approach.
                </p>
                <p className="text-xl text-gray-700 leading-relaxed">
                  By integrating <span className="font-semibold text-teal-600">physiological signals</span>, <span className="font-semibold text-purple-600">voice-based emotional cues</span>, and <span className="font-semibold text-indigo-600">self-reported assessments</span>, SafeSpace captures a fuller picture of stress than traditional single-modality systems.
                </p>
              </div>
              
              {/* Key Benefits */}
              <div className="grid grid-cols-2 gap-4">
                <div className="p-4 bg-blue-50 rounded-xl border border-blue-200">
                  <Shield className="w-8 h-8 text-blue-600 mb-2" />
                  <h4 className="font-bold text-gray-900 mb-1">Privacy First</h4>
                  <p className="text-sm text-gray-600">Edge processing ensures data security</p>
                </div>
                <div className="p-4 bg-teal-50 rounded-xl border border-teal-200">
                  <Zap className="w-8 h-8 text-teal-600 mb-2" />
                  <h4 className="font-bold text-gray-900 mb-1">Real-time</h4>
                  <p className="text-sm text-gray-600">Instant stress level detection</p>
                </div>
              </div>
            </AnimatedCard>
            
            <AnimatedCard data-animate="hero-visual" delay={200}>
              <div className="relative">
                <div className="absolute -inset-4 bg-gradient-to-r from-blue-500/20 to-purple-500/20 rounded-3xl blur-2xl"></div>
                <div className="relative bg-gradient-to-br from-white to-blue-50/50 rounded-3xl p-12 shadow-2xl border border-white/50 backdrop-blur-sm">
                  <div className="text-center">
                    <div className="relative mb-8">
                      <div className="absolute inset-0 bg-gradient-to-r from-blue-500/20 to-purple-500/20 rounded-full blur-xl"></div>
                      <Activity className="relative w-24 h-24 mx-auto text-blue-600" />
                    </div>
                    <h3 className="text-2xl font-bold text-gray-900 mb-3">Multimodal Analysis</h3>
                    <p className="text-gray-600 mb-6">Real-time processing of multiple data streams</p>
                    <div className="flex justify-center space-x-4">
                      <div className="w-3 h-3 bg-blue-500 rounded-full animate-pulse"></div>
                      <div className="w-3 h-3 bg-teal-500 rounded-full animate-pulse" style={{animationDelay: '0.5s'}}></div>
                      <div className="w-3 h-3 bg-purple-500 rounded-full animate-pulse" style={{animationDelay: '1s'}}></div>
                    </div>
                  </div>
                </div>
              </div>
            </AnimatedCard>
          </div>
        </div>

        {/* Enhanced Multimodal Approach */}
        <div className="mb-24">
          <AnimatedCard data-animate="approach-title" className="text-center mb-16">
            <h3 className="text-4xl sm:text-5xl font-bold text-gray-900 mb-6">
              Our Multimodal Approach
            </h3>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto">
              Three powerful data streams working in harmony for unprecedented accuracy
            </p>
          </AnimatedCard>
          
          <div className="grid gap-8 md:grid-cols-2 lg:grid-cols-3">
            <FeatureCard
              icon={Activity}
              title="Physiological Signals"
              description="WESAD dataset integration with EDA, ECG, temperature, and accelerometry for comprehensive autonomic arousal detection and real-time biosignal processing."
              badge="Real-time biosignal processing"
              gradient="from-blue-400/90 to-blue-600/90"
              delay={0}
            />
            <FeatureCard
              icon={Mic}
              title="Voice Analysis"
              description="RAVDESS and IEMOCAP datasets for advanced emotional cue detection through sophisticated pitch jitter analysis and harmonic pattern recognition."
              badge="Advanced speech processing"
              gradient="from-teal-400/90 to-teal-600/90"
              delay={150}
            />
            <FeatureCard
              icon={FileText}
              title="Self-Assessment"
              description="DASS-21 questionnaire integration for capturing subjective stress experiences and psychological context with clinical validation standards."
              badge="Clinical validation"
              gradient="from-purple-400/90 to-purple-600/90"
              delay={300}
            />
          </div>
        </div>

        {/* Enhanced XAI Section */}
        <div className="mb-24">
          <div className="grid lg:grid-cols-2 gap-16 items-center">
            <AnimatedCard data-animate="xai-visual" className="order-2 lg:order-1">
              <div className="relative">
                <div className="absolute -inset-6 bg-gradient-to-r from-purple-500/20 to-blue-500/20 rounded-3xl blur-2xl"></div>
                <div className="relative bg-gradient-to-br from-white to-purple-50/50 rounded-3xl p-12 shadow-2xl border border-white/50 backdrop-blur-sm">
                  <div className="text-center">
                    <Brain className="w-24 h-24 mx-auto text-purple-600 mb-6" />
                    <h4 className="text-2xl font-bold text-gray-900 mb-3">Explainable AI</h4>
                    <p className="text-gray-600 mb-6">Transparent decision-making process</p>
                    <div className="space-y-3">
                      <div className="flex items-center justify-between bg-white/80 p-3 rounded-xl">
                        <span className="text-sm font-medium text-gray-700">SHAP Analysis</span>
                        <div className="w-20 h-2 bg-blue-200 rounded-full overflow-hidden">
                          <div className="w-4/5 h-full bg-blue-500 rounded-full"></div>
                        </div>
                      </div>
                      <div className="flex items-center justify-between bg-white/80 p-3 rounded-xl">
                        <span className="text-sm font-medium text-gray-700">LIME Integration</span>
                        <div className="w-20 h-2 bg-teal-200 rounded-full overflow-hidden">
                          <div className="w-3/4 h-full bg-teal-500 rounded-full"></div>
                        </div>
                      </div>
                      <div className="flex items-center justify-between bg-white/80 p-3 rounded-xl">
                        <span className="text-sm font-medium text-gray-700">Gradients</span>
                        <div className="w-20 h-2 bg-purple-200 rounded-full overflow-hidden">
                          <div className="w-4/5 h-full bg-purple-500 rounded-full"></div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </AnimatedCard>
            
            <AnimatedCard data-animate="xai-content" delay={200} className="space-y-8 order-1 lg:order-2">
              <div>
                <div className="inline-block p-2 bg-purple-100 rounded-lg mb-4">
                  <Brain className="w-6 h-6 text-purple-600" />
                </div>
                <h3 className="text-4xl font-bold text-gray-900 mb-6">
                  Explainable AI Integration
                </h3>
                <p className="text-xl text-gray-700 leading-relaxed mb-8">
                  A central goal of SafeSpace is not only to apply explainable AI (XAI), but to compare techniques such as 
                  <span className="font-bold text-purple-600"> SHAP, LIME, and Integrated Gradients</span> to understand which 
                  best aligns with our system structure.
                </p>
              </div>
              
              <div className="space-y-4">
                <div className="group p-6 bg-gradient-to-r from-blue-50 to-blue-100/50 rounded-2xl border border-blue-200 hover:shadow-lg transition-all duration-300">
                  <div className="flex items-start space-x-4">
                    <div className="w-3 h-3 bg-blue-500 rounded-full mt-2 flex-shrink-0 group-hover:scale-125 transition-transform duration-300"></div>
                    <div>
                      <h4 className="font-bold text-gray-900 mb-2">SHAP (SHapley Additive exPlanations)</h4>
                      <p className="text-gray-700">Advanced feature importance analysis with game-theoretic foundations</p>
                    </div>
                  </div>
                </div>
                
                <div className="group p-6 bg-gradient-to-r from-teal-50 to-teal-100/50 rounded-2xl border border-teal-200 hover:shadow-lg transition-all duration-300">
                  <div className="flex items-start space-x-4">
                    <div className="w-3 h-3 bg-teal-500 rounded-full mt-2 flex-shrink-0 group-hover:scale-125 transition-transform duration-300"></div>
                    <div>
                      <h4 className="font-bold text-gray-900 mb-2">LIME</h4>
                      <p className="text-gray-700">Local interpretable model explanations for transparent decision-making</p>
                    </div>
                  </div>
                </div>
                
                <div className="group p-6 bg-gradient-to-r from-purple-50 to-purple-100/50 rounded-2xl border border-purple-200 hover:shadow-lg transition-all duration-300">
                  <div className="flex items-start space-x-4">
                    <div className="w-3 h-3 bg-purple-500 rounded-full mt-2 flex-shrink-0 group-hover:scale-125 transition-transform duration-300"></div>
                    <div>
                      <h4 className="font-bold text-gray-900 mb-2">Integrated Gradients</h4>
                      <p className="text-gray-700">Deep learning interpretability through attribution methods</p>
                    </div>
                  </div>
                </div>
              </div>
            </AnimatedCard>
          </div>
        </div>

        {/* Enhanced Edge Computing Section */}
        {/* <div className="mb-24">
          <div className="grid lg:grid-cols-2 gap-16 items-center">
            <AnimatedCard data-animate="edge-content" className="space-y-8">
              <div>
                <div className="inline-block p-2 bg-green-100 rounded-lg mb-4">
                  <Cpu className="w-6 h-6 text-green-600" />
                </div>
                <h3 className="text-4xl font-bold text-gray-900 mb-6">
                  Edge Computing & Real-time Processing
                </h3>
                <p className="text-xl text-gray-700 leading-relaxed mb-8">
                  Designed for real-world deployment, SafeSpace runs on an 
                  <span className="font-bold text-green-600"> ESP32-based edge device</span> with cloud connectivity via 
                  Firebase, enabling low-latency, secure, and scalable deployment.
                </p>
              </div>
              
              <div className="bg-gradient-to-br from-green-50 to-emerald-50 p-8 rounded-3xl border border-green-200/50 backdrop-blur-sm">
                <h4 className="font-bold text-gray-900 mb-6 text-xl flex items-center">
                  <Award className="w-6 h-6 text-green-600 mr-3" />
                  Technical Specifications
                </h4>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  <TechSpec icon={Cpu} text="ESP32 microcontroller" delay={0} />
                  <TechSpec icon={Shield} text="Firebase cloud integration" delay={100} />
                  <TechSpec icon={Zap} text="Low-latency inference" delay={200} />
                  <TechSpec icon={Users} text="Enhanced privacy protection" delay={300} />
                  <TechSpec icon={Activity} text="Real-time data processing" delay={400} />
                  <TechSpec icon={TrendingUp} text="Scalable architecture" delay={500} />
                </div>
              </div>
            </AnimatedCard>
            
            <AnimatedCard data-animate="edge-visual" delay={200}>
              <div className="relative">
                <div className="absolute -inset-6 bg-gradient-to-r from-green-500/20 to-emerald-500/20 rounded-3xl blur-2xl"></div>
                <div className="relative bg-gradient-to-br from-white to-green-50/50 rounded-3xl p-12 shadow-2xl border border-white/50 backdrop-blur-sm">
                  <div className="text-center">
                    <div className="relative mb-8">
                      <div className="absolute inset-0 bg-gradient-to-r from-green-500/20 to-emerald-500/20 rounded-full blur-xl"></div>
                      <Cpu className="relative w-24 h-24 mx-auto text-green-600" />
                    </div>
                    <h4 className="text-2xl font-bold text-gray-900 mb-3">ESP32 Edge Device</h4>
                    <p className="text-gray-600 mb-6">Powerful embedded processing</p>
                    <div className="grid grid-cols-2 gap-4 text-sm">
                      <div className="bg-white/80 p-3 rounded-xl">
                        <div className="font-bold text-green-600">240MHz</div>
                        <div className="text-gray-600">Dual Core</div>
                      </div>
                      <div className="bg-white/80 p-3 rounded-xl">
                        <div className="font-bold text-green-600">WiFi/BT</div>
                        <div className="text-gray-600">Connectivity</div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </AnimatedCard>
          </div>
        </div> */}

        {/* Enhanced Research Impact Section */}
        <AnimatedCard data-animate="research-impact">
          <div className="relative bg-gradient-to-br from-blue-800 via-indigo-900 to-purple-900 rounded-3xl p-0 shadow-2xl overflow-visible max-w-2xl mx-auto border border-blue-900/40">
            {/* Decorative Glow Ring */}
            <div className="absolute -top-8 left-1/2 -translate-x-1/2 w-40 h-40 bg-gradient-to-tr from-blue-400/40 via-purple-400/30 to-indigo-400/40 rounded-full blur-2xl opacity-70 z-0"></div>
            {/* Card Content */}
            <div className="relative z-10 rounded-3xl bg-white/5 backdrop-blur-xl p-6 sm:p-10 flex flex-col items-center">
              <div className="-mt-14 mb-2 flex items-center justify-center w-20 h-20 rounded-full bg-gradient-to-tr from-blue-500 via-purple-500 to-indigo-500 shadow-lg border-4 border-white/20">
                <Award className="w-10 h-10 text-white drop-shadow-lg" />
              </div>
              <h3 className="text-2xl sm:text-3xl font-extrabold text-white mb-2 text-center tracking-tight drop-shadow-lg">
                Research Impact & Innovation
              </h3>
              <p className="text-base sm:text-lg text-blue-100 max-w-lg mx-auto leading-relaxed text-center mb-6">
                Integrating multimodal data and XAI increases both accuracy and trust. SafeSpace is a leap toward stress detection that is as <span className="font-semibold text-blue-300">understandable</span> as it is <span className="font-semibold text-purple-200">effective</span>.
              </p>
              <div className="flex flex-col sm:flex-row gap-3 w-full justify-center items-center">
                <StatCard
                  value="95%+"
                  label="Detection Accuracy"
                  icon={Target}
                  color="bg-gradient-to-r from-blue-500 to-blue-600"
                  delay={0}
                />
                <StatCard
                  value="<100ms"
                  label="Response Latency"
                  icon={Zap}
                  color="bg-gradient-to-r from-teal-500 to-teal-600"
                  delay={150}
                />
                <StatCard
                  value="3"
                  label="Modalities Integrated"
                  icon={Activity}
                  color="bg-gradient-to-r from-purple-500 to-purple-600"
                  delay={300}
                />
              </div>
            </div>
            {/* Subtle bottom glow */}
            <div className="absolute -bottom-8 left-1/2 -translate-x-1/2 w-32 h-10 bg-gradient-to-r from-blue-400/30 via-purple-400/30 to-indigo-400/30 rounded-full blur-2xl opacity-60 z-0"></div>
          </div>
        </AnimatedCard>
      </div>
    </section>
  )
}
