import React, { useState, useEffect } from 'react';
import { motion, useInView } from 'framer-motion';

const FluidStat = ({ label, targetValue, suffix, delay }) => {
  const ref = React.useRef(null);
  const isInView = useInView(ref, { once: true, margin: "-10%" });
  const [value, setValue] = useState(0);

  useEffect(() => {
    if (isInView) {
      let start = 0;
      const end = targetValue;
      const duration = 2000;
      const incrementTime = 20;
      const step = (end / (duration / incrementTime));

      const timer = setInterval(() => {
        start += step;
        if (start >= end) {
          setValue(end);
          clearInterval(timer);
        } else {
          setValue(start);
        }
      }, incrementTime);
      return () => clearInterval(timer);
    }
  }, [isInView, targetValue]);

  return (
    <motion.div 
      ref={ref}
      className="fluid-stat"
      initial={{ opacity: 0, y: 30, filter: 'blur(10px)' }}
      animate={isInView ? { opacity: 1, y: 0, filter: 'blur(0px)' } : {}}
      transition={{ duration: 1.5, delay, ease: "easeOut" }}
    >
      <div className="stat-value">
        {value.toFixed(suffix === '%' ? 1 : 2)}{suffix}
      </div>
      <div className="stat-label">{label}</div>
    </motion.div>
  );
};

export default function StatsSection() {
  return (
    <div className="tree-branch" style={{ flexDirection: 'column', alignItems: 'center', marginBottom: '30vh' }}>
      <motion.div
        initial={{ opacity: 0, filter: 'blur(10px)' }}
        whileInView={{ opacity: 1, filter: 'blur(0px)' }}
        viewport={{ once: true, margin: "-10%" }}
        transition={{ duration: 1.5 }}
        style={{ textAlign: 'center', marginBottom: '4rem' }}
      >
        <h2 className="gradient-text">The Results</h2>
        <p style={{ margin: '0 auto' }}>
          Evaluated on the held-out Symptom2Disease dataset (1,200 records, 24 classes).
        </p>
      </motion.div>
      
      <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
        <FluidStat label="Accuracy" targetValue={87.5} suffix="%" delay={0.2} />
        <FluidStat label="Macro-F1 Score" targetValue={0.86} suffix="" delay={0.4} />
        <FluidStat label="Unparseable Outputs" targetValue={0.0} suffix="%" delay={0.6} />
      </div>
    </div>
  );
}
