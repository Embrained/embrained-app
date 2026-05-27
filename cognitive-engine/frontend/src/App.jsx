/*
 * Embrained - Neural Navigation Software Suite
 * Copyright (C) 2026 Embrained
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

import React from 'react';
import { useRobotConnection } from './hooks/useRobotConnection';
import LiveMode from './components/LiveMode';

function App() {
  // Determine WS URL (use window.location to support running on other devices)
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const host = window.location.hostname;
  // backend port is 8000. If we are serving via python, python serves frontend on 8000 too.
  // If dev mode (vite), frontend is 5173, backend is 8000.
  const wsUrl = `${protocol}//${host}:8000/ws`;

  const { data, history, connected, sendMessage } = useRobotConnection(wsUrl);

  return (
    <LiveMode
      data={data}
      history={history}
      connected={connected}
      sendMessage={sendMessage}
    />
  );
}

export default App;
