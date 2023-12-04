import { Outlet } from 'react-router-dom';
import MetaMask from '../components/MetaMask';

export default function Root() {
    return (
        <>
            <div id="sidebar">
                <nav>
                    <a href={'/'}><h1>Blockchain Bazaar</h1></a>
                    <ul>
                        <li>
                            <a href={'/shop'}>Shop</a>
                        </li>
                    </ul>
                </nav>
                <MetaMask />
            </div>
            <div id="detail">
                <div id="header">
                    <h1><img src="/logo_coin.png" alt="logo"/>Blockchain Bazaar</h1>
                </div>
                <Outlet />
            </div>
        </>
    )
}