import React, { useState, useEffect, useCallback, useRef } from 'react';
import './GameOfLife.css';

interface Cell {
  active: boolean;
  intensity: number;
  age: number;
  energy: number;
}

interface Rules {
  survive: number[];
  birth: number[];
}

type RuleSetName = 'conway' | 'highlife' | 'daynight' | 'seeds' | 'maze' | 'coral';

const GameOfLife: React.FC = () => {
  const GRID_SIZE = 16;
  const RULE_SETS: Record<RuleSetName, Rules> = {
    conway: { survive: [2, 3], birth: [3] }, // Standard Conway's Game of Life rules
    highlife: { survive: [2, 3], birth: [3, 6] },
    daynight: { survive: [3, 4, 6, 7, 8], birth: [3, 6, 7, 8] },
    seeds: { survive: [], birth: [2] },
    maze: { survive: [1, 2, 3, 4, 5], birth: [3] },
    coral: { survive: [4, 5, 6, 7, 8], birth: [3] }
  };

  const [grid, setGrid] = useState<Cell[][]>([]);
  const [isVisible, setIsVisible] = useState(true);
  const animationRef = useRef<number | null>(null);
  const lastUpdateRef = useRef<number>(0);
  
  // Use a more dynamic ruleset
  const [currentRuleSet] = useState<{name: RuleSetName; rules: Rules}>({
    name: 'highlife',
    rules: RULE_SETS['highlife']
  });

  const createCell = (active: boolean): Cell => ({
    active,
    intensity: active ? 0.7 : 0,
    age: 0,
    energy: Math.random() * 0.7
  });

  const initializeGrid = useCallback(() => {
    const density = 0.4; // Higher density for more activity
    
    const initialGrid = Array(GRID_SIZE).fill(null).map(() => 
      Array(GRID_SIZE).fill(null).map(() => createCell(Math.random() < density))
    );
    setGrid(initialGrid);
  }, []);

  useEffect(() => {
    // Initialize the grid when the component mounts
    initializeGrid();
    
    const observer = new IntersectionObserver(
      ([entry]) => {
        setIsVisible(entry.isIntersecting);
      },
      { threshold: 0.1 }
    );

    if (containerRef.current) {
      observer.observe(containerRef.current);
    }

    return () => {
      observer.disconnect();
      if (animationRef.current !== null) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [initializeGrid]);

  const containerRef = useRef<HTMLDivElement>(null);

  const applySymmetry = useCallback((newGrid: Cell[][]) => {
    const mid = Math.floor(GRID_SIZE / 2);
    for (let y = 0; y < mid; y++) {
      for (let x = 0; x < mid; x++) {
        newGrid[y][GRID_SIZE-1-x] = {...newGrid[y][x]};
        newGrid[GRID_SIZE-1-y][x] = {...newGrid[y][x]};
        newGrid[GRID_SIZE-1-y][GRID_SIZE-1-x] = {...newGrid[y][x]};
      }
    }
    return newGrid;
  }, []);

  const getNeighbors = useCallback((grid: Cell[][], x: number, y: number) => {
    let count = 0;
    for (let i = -1; i <= 1; i++) {
      for (let j = -1; j <= 1; j++) {
        if (i === 0 && j === 0) continue;
        const newX = x + i;
        const newY = y + j;
        if (newX >= 0 && newX < GRID_SIZE && newY >= 0 && newY < GRID_SIZE) {
          if (grid[newY][newX].active) {
            count++;
          }
        }
      }
    }
    return count;
  }, []);

  const updateGrid = useCallback(() => {
    setGrid(prevGrid => {
      let newGrid = prevGrid.map((row, y) => 
        row.map((cell, x) => {
          const neighbors = getNeighbors(prevGrid, x, y);
          let nextActive = cell.active;
          
          if (cell.active) {
            nextActive = currentRuleSet.rules.survive.includes(neighbors);
          } else {
            nextActive = currentRuleSet.rules.birth.includes(neighbors);
          }

          // Add random activation chance
          if (!nextActive && Math.random() < 0.001) {
            nextActive = true;
          }

          // Calculate wave effect for some movement
          const time = Date.now() * 0.0003;
          const waveEffect = Math.sin(time + (x + y) * 0.1) * 0.05;
          const xDiff = x - GRID_SIZE/2;
          const yDiff = y - GRID_SIZE/2;
          const distanceToCenter = Math.sqrt(xDiff * xDiff + yDiff * yDiff) / (GRID_SIZE * 0.707);
          
          // Calculate intensity changes
          let nextIntensity = 0;
          if (nextActive) {
            if (!cell.active) {
              nextIntensity = 0.2; // Start at visible intensity when activated
            } else {
              nextIntensity = Math.min(0.7, cell.intensity + 0.1); // Faster ramp up
            }
          } else if (cell.active) {
            nextIntensity = Math.max(0, cell.intensity - 0.1); // Faster fade out
            if (nextIntensity < 0.1) nextIntensity = 0;
          }
          
          const nextEnergy = Math.max(0, Math.min(0.7,
            (cell.energy * 0.95) + (nextActive ? 0.05 : -0.05) +
            (waveEffect * (1 - distanceToCenter))
          ));

          return {
            active: nextActive || nextIntensity > 0,
            intensity: nextIntensity,
            age: nextActive ? (cell.active ? cell.age + 1 : 0) : 0,
            energy: nextEnergy
          };
        })
      );

      // Check if we need to reset - but do it rarely
      let activeCount = 0;
      for (let y = 0; y < GRID_SIZE; y += 2) {
        for (let x = 0; x < GRID_SIZE; x += 2) {
          if (newGrid[y][x].active) activeCount++;
        }
      }
      
      // If almost all cells are active or almost none are, reset
      const cellsChecked = (GRID_SIZE/2) * (GRID_SIZE/2);
      if (activeCount > cellsChecked * 0.9 || activeCount < cellsChecked * 0.05) {
        return Array(GRID_SIZE).fill(null).map(() => 
          Array(GRID_SIZE).fill(null).map(() => createCell(Math.random() < 0.4))
        );
      }

      return applySymmetry(newGrid);
    });
  }, [applySymmetry, getNeighbors, currentRuleSet.rules]);

  // Animation loop with requestAnimationFrame
  useEffect(() => {
    if (!isVisible) {
      if (animationRef.current !== null) {
        cancelAnimationFrame(animationRef.current);
        animationRef.current = null;
      }
      return;
    }
    
    const animate = (timestamp: number) => {
      if (timestamp - lastUpdateRef.current >= 800) { // Update every 800ms
        lastUpdateRef.current = timestamp;
        updateGrid();
      }
      
      animationRef.current = requestAnimationFrame(animate);
    };
    
    animationRef.current = requestAnimationFrame(animate);
    
    return () => {
      if (animationRef.current !== null) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [isVisible, updateGrid]);

  const renderCells = useCallback(() => {
    return grid.map((row, y) => (
      <div key={y} className="flex">
        {row.map((cell, x) => {
          const edgeFade = Math.min(x, y, GRID_SIZE - x - 1, GRID_SIZE - y - 1) / (GRID_SIZE / 2);
          const isActive = cell.active;
          
          const opacityMultiplier = 1.0;
          const bgColor = isActive 
            ? `rgba(23, 23, 23, ${Math.min(0.3, cell.intensity * edgeFade * opacityMultiplier)})` 
            : `rgba(255, 255, 255, ${Math.min(0.3, cell.intensity * edgeFade * opacityMultiplier)})`;
          
          return (
            <div
              key={`${x}-${y}`}
              className="cell"
              style={{ 
                backgroundColor: bgColor,
                width: '80px',
                height: '80px'
              }}
            />
          );
        })}
      </div>
    ));
  }, [grid]);

  return (
    <div ref={containerRef} className="game-of-life-container">
      <div className="grid-container">
        <div className="grid">
          {renderCells()}
        </div>
      </div>
    </div>
  );
};

export default GameOfLife; 