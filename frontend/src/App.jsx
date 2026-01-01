import React, { useState } from 'react';
import { useRobotConnection } from './hooks/useRobotConnection';
import Home from './components/Home';
import LiveMode from './components/LiveMode';
import TrainingMode from './components/TrainingMode';

function App() {
  // Determine WS URL (use window.location to support running on other devices)
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const host = window.location.hostname;
  // backend port is 8000. If we are serving via python, python serves frontend on 8000 too.
  // If dev mode (vite), frontend is 5173, backend is 8000.
  const wsUrl = `${protocol}//${host}:8000/ws`;

  // Connection Hook lives at top level to maintain connection across view changes?
  // Or should we reconnect when entering modes? 
  // User wanted "Home" which implies disconnected state? Or connected but idle?
  // Let's keep connection alive for now to be responsive.
  const { data, history, connected, sendMessage } = useRobotConnection(wsUrl);

  const [currentView, setCurrentView] = useState('HOME'); // HOME, LIVE, INFERENCE, TRAINING

  const handleModeSelect = (mode) => {
    setCurrentView(mode);
    if (mode === 'LIVE' || mode === 'INFERENCE' || mode === 'TRAINING') {
      // Tell backend to switch mode if applicable (Wait, Training might be purely frontend?)
      // Actually backend engine has 'LIVE' and 'INFERENCE'. Training is separate.
      if (mode !== 'TRAINING') {
        sendMessage('SET_MODE', mode);
      }
    }
  };

  const handleExit = () => {
    setCurrentView('HOME');
    // Force backend to Idle/Manual (LIVE) to stop autonomous planner
    sendMessage('SET_MODE', 'IDLE');
    sendMessage('MOVE', 3); // Stop
  };

  return (
    <>
      {currentView === 'HOME' && <Home onSelectMode={handleModeSelect} />}

      {currentView === 'LIVE' && (
        <LiveMode
          data={data}
          history={history}
          sendMessage={sendMessage}
          onExit={handleExit}
        />
      )}

      {currentView === 'TRAINING' && <TrainingMode data={data} onExit={handleExit} />}
    </>
  );
}


export default App;
