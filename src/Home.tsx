import React from 'react';
import GameOfLife from './components/GameOfLife';
import './Home.css';

const Home: React.FC = () => {
  const openPaper = () => {
    window.open('https://www.nature.com/articles/s42256-025-01031-9', '_blank');
  };
  
  const openWebApp = () => {
    window.open('https://simg.cheme.cmu.edu', '_blank');
  };
  
  return (
    <div className="home-container">
      <div className="title-container">
        <div className="title-wrapper">
          <h1 className="title">SIMG</h1>
          <div className="subtitle-wrapper">
            <h2 className="subtitle">Stereoelectronics-Infused</h2>
            <h2 className="subtitle">Molecular Graphs</h2>
          </div>
        </div>
      </div>
      
      <div className="paper-container">
        <h3 className="paper-title">
          Advancing molecular machine learning representations with stereoelectronics-infused molecular graphs
        </h3>
        
        <p className="paper-authors">
          Daniil A. Boiko, Thiago Reschützegger, Benjamin Sanchez-Lengeling, Samuel M. Blau & Gabe Gomes
        </p>
        
        <div className="abstract-container">
          <h4 className="abstract-title">Abstract</h4>
          <p className="abstract-text">
            Molecular representation is a critical element in our understanding of the physical world and the foundation for modern molecular machine learning. Previous molecular machine learning models have used strings, fingerprints, global features and simple molecular graphs that are inherently information-sparse representations. However, as the complexity of prediction tasks increases, the molecular representation needs to encode higher fidelity information. This work introduces a new approach to infusing quantum-chemical-rich information into molecular graphs via stereoelectronic effects, enhancing expressivity and interpretability. Learning to predict the stereoelectronics-infused representation with a tailored double graph neural network workflow enables its application to any downstream molecular machine learning task without expensive quantum-chemical calculations. We show that the explicit addition of stereoelectronic information substantially improves the performance of message-passing two-dimensional machine learning models for molecular property prediction. We show that the learned representations trained on small molecules can accurately extrapolate to much larger molecular structures, yielding chemical insight into orbital interactions for previously intractable systems, such as entire proteins, opening new avenues of molecular design. Finally, we have developed a web application (simg.cheme.cmu.edu) where users can rapidly explore stereoelectronic information for their own molecular systems.
          </p>
        </div>
      </div>
      
      <div className="action-buttons">
        <button onClick={openPaper} className="action-button primary">
          Read the Paper
        </button>
        <button onClick={openWebApp} className="action-button secondary">
          Try the Web App
        </button>
      </div>
      
      <GameOfLife />
    </div>
  );
};

export default Home; 