import React from 'react';
import ReactDOM from 'react-dom/client';
import { createBrowserRouter, RouterProvider } from 'react-router-dom';

import Root from './routes/root';
import ErrorPage from './routes/error-page';
import Shop from './routes/shop';
import Items from './routes/items';

import './index.css';

const router = createBrowserRouter([
  {
    path: "/",
    element: <Root />,
    errorElement: <ErrorPage />,
    children: [
      { index: true, element: <Home /> },
      { path: "/shop", element: <Shop /> },
      { path: "/items", element: <Items /> }
    ]
  }
])

function Home() {
  return (
      <div>
          <body>
            <p>
              Blockchain Bazaar is an AI controlled video game item shop. 
              Currently, video game economies typically fall into two categories: player shops and non-player character (NPC) shops.
            </p>
            <p>  
              In player shops, item prices, quantities, and value are dynamic and change based on free market forces. 
              However, player shops can be plagued by dishonesty and price gouging. In NPC shops, prices are static and non-interactive. 
              Apart from a lack of immersion , the flaw here is that the value of an item is irrelevant if the shops can provide an unlimited supply of it and inevitably cause the inflation of in-game currency. 
            </p>
            <p>
              Blockchain Bazaar aims to take the best of both systems by using blockchain to secure trades and maintain unique items while AI is used to alter prices based on user transactions.
            </p>
          </body>
          <h2>Our Team</h2>
          <ul>
            <li>Lorenzo Mercado</li>
            <li>Justin Chang</li>
            <li>Konain Qureshi</li>
          </ul>
      </div>
  )
}

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <RouterProvider router={router} />
  </React.StrictMode>
)
