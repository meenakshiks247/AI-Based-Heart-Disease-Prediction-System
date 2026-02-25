import React, { useState, useRef } from 'react'
import Navbar from './components/Navbar'
import Hero from './components/Hero'
import PredictionForm from './components/PredictionForm'
import ResultCard from './components/ResultCard'

export default function App() {
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const formRef = useRef(null)

  const scrollToPredict = () => {
    formRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' })
  }

  return (
    <>
      {/* Global reset + dark theme */}
      <style>{`
        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
        html { scroll-behavior: smooth; }
        body {
          font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif;
          background: #0a0a1e;
          min-height: 100vh;
          color: #e0e0e0;
          overflow-x: hidden;
        }
        ::selection { background: rgba(127,90,240,0.4); }
        input::-webkit-outer-spin-button,
        input::-webkit-inner-spin-button { -webkit-appearance: none; margin: 0; }
        input[type=number] { -moz-appearance: textfield; }
      `}</style>

      <Navbar onPredictClick={scrollToPredict} />
      <Hero onStartCheck={scrollToPredict} />
      <div style={{ background: '#0e0e24' }}>
        <PredictionForm ref={formRef} onResult={setResult} onError={setError} />
        <ResultCard result={result} />
        {error && (
          <div style={{
            maxWidth: 820,
            margin: '0 auto',
            padding: '0 28px 40px',
          }}>
            <div style={{
              padding: '14px 20px',
              borderRadius: 10,
              background: 'rgba(255,69,58,0.12)',
              border: '1px solid rgba(255,69,58,0.35)',
              color: '#ff6b6b',
              textAlign: 'center',
              fontWeight: 500,
            }}>
              {error}
            </div>
          </div>
        )}
      </div>

      {/* Footer */}
      <footer style={{
        background: '#08081a',
        borderTop: '1px solid rgba(255,255,255,0.06)',
        padding: '32px 28px',
        textAlign: 'center',
        color: '#706f8a',
        fontSize: '0.82rem',
      }}>
        <p style={{ marginBottom: 4 }}>
          <span style={{ fontWeight: 700, color: '#a78bfa' }}>Heartify AI</span> — AI-Powered Heart Disease Prediction System
        </p>
        <p>© {new Date().getFullYear()} All rights reserved. For educational and screening purposes only.</p>
      </footer>
    </>
  )
}
