# StressBuster Games - SafeSpace AI

## Overview
StressBuster is a collection of stress-relieving games integrated into the SafeSpace AI platform. These games provide users with quick mental breaks and cognitive training while they wait for stress analysis results or simply need a moment to relax.

## Features

### 🦕 Dinosaur Jump Game
- **Type**: Endless runner game
- **Duration**: 2-5 minutes
- **Benefits**: Improves reflexes, focus, and provides instant stress relief
- **Controls**: 
  - Click or press Spacebar to jump
  - Avoid obstacles (cacti) to survive longer
  - High score tracking with localStorage persistence

### 🧠 Memory Challenge Game
- **Type**: Card matching game
- **Duration**: 3-7 minutes
- **Benefits**: Enhances memory, logic, and cognitive function
- **Features**:
  - Three difficulty levels (Easy: 6 pairs, Medium: 8 pairs, Hard: 12 pairs)
  - Performance rating system with star ratings
  - Move counter and timer
  - Victory celebration modal

## Technical Implementation

### Components
- `DinosaurGame.tsx` - Canvas-based game with physics and collision detection
- `MemoryGame.tsx` - React-based card matching game with state management
- `StressBusterPage.tsx` - Main page with game selection and navigation

### Game Features
- **Responsive Design**: Works on desktop and mobile devices
- **Smooth Animations**: GSAP animations and CSS transitions
- **Local Storage**: High scores and game progress persistence
- **Accessibility**: Keyboard and mouse/touch controls

### Navigation Integration
- Added to main navbar (desktop and mobile)
- Featured button on homepage hero section
- Cross-linking between stress analysis and games
- Seamless navigation between different sections

## Usage

### For Users
1. Navigate to StressBuster from the navbar or homepage
2. Choose between Dinosaur Jump or Memory Challenge
3. Play games for quick stress relief
4. Return to stress analysis when ready

### For Developers
1. Games are located in `/app/stress-buster/`
2. Game components are in `/app/components/`
3. Easy to add new games by extending the existing structure
4. All games follow the same design patterns and styling

## Benefits

### Mental Health
- **Immediate Relief**: Quick 2-7 minute sessions
- **Cognitive Training**: Improves memory and focus
- **Stress Reduction**: Engaging gameplay distracts from stressors
- **Mood Enhancement**: Fun, colorful interface boosts spirits

### User Experience
- **Seamless Integration**: Part of the main SafeSpace AI platform
- **Quick Access**: Easy navigation from any page
- **Progress Tracking**: High scores and performance metrics
- **Responsive Design**: Works on all devices

## Future Enhancements
- Additional game types (puzzle, rhythm, etc.)
- User progress tracking across sessions
- Social features (leaderboards, achievements)
- Integration with stress analysis results
- Personalized game recommendations

## Technical Notes
- Built with Next.js 13+ and React 18
- Uses Canvas API for Dinosaur game
- Implements modern React patterns (hooks, context)
- Responsive design with Tailwind CSS
- GSAP animations for smooth transitions
