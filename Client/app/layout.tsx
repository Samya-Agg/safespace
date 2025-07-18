import type React from "react"
import type { Metadata } from "next"
import { Inter } from "next/font/google"
import "./globals.css"
import Navbar from "./components/Navbar"
const inter = Inter({ subsets: ["latin"] })

export const metadata: Metadata = {
  title: "SafeSpace AI - Mental Health Empowered by AI",
  description: "Detect stress through voice, sensors, and psychology with SafeSpace AI",
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <script async={false} src="https://cdn.botpress.cloud/webchat/v3.0/inject.js"></script>
      <script async={false} src="https://files.bpcontent.cloud/2025/07/11/16/20250711160316-693TLH0U.js"></script>
      <body className={inter.className}>
        <Navbar />
        {children}
      </body>
    </html>
  )
}
