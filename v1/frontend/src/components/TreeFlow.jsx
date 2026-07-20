import React, { useRef } from 'react';
import { motion, useScroll, useSpring } from 'framer-motion';

export default function TreeFlow({ children }) {
  const containerRef = useRef(null);
  
  const { scrollYProgress } = useScroll({
    target: containerRef,
    offset: ["start start", "end end"]
  });

  const smoothProgress = useSpring(scrollYProgress, {
    stiffness: 100,
    damping: 30,
    restDelta: 0.001
  });

  return (
    <div ref={containerRef} className="tree-container">
      <svg 
        style={{
          position: 'absolute',
          top: 0,
          left: '50%',
          width: '4px',
          height: '100%',
          transform: 'translateX(-50%)',
          zIndex: -1,
          overflow: 'visible'
        }}
      >
        <motion.line
          x1="2"
          y1="0"
          x2="2"
          y2="100%"
          stroke="url(#liquid-gradient)"
          strokeWidth="4"
          strokeLinecap="round"
          style={{ pathLength: smoothProgress }}
        />
        <defs>
          <linearGradient id="liquid-gradient" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#ffffff" stopOpacity="1" />
            <stop offset="30%" stopColor="#76b900" stopOpacity="1" />
            <stop offset="70%" stopColor="#76b900" stopOpacity="0.8" />
            <stop offset="100%" stopColor="#000000" stopOpacity="0" />
          </linearGradient>
        </defs>
      </svg>
      
      <div style={{ width: '100%', position: 'relative', zIndex: 1 }}>
        {children}
      </div>
    </div>
  );
}
