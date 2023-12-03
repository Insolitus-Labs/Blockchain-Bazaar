import { useEffect } from 'react';
import { useState } from 'react';
import { ethers } from 'ethers';

import contract from '../contracts/contract.json';

const contractAddress = '0x2fF18602C615b408d182FEE8591ad87968d1c644';
const abi = contract.abi;

export default function MetaMask() {

  const [currentAccount, setCurrentAccount] = useState(null);
  
  useEffect(() => {
    checkWalletIsConnected();
  }, [])

  const checkWalletIsConnected = async () => {
    const { ethereum } = window;

    if (!ethereum) {
      console.log("Make sure to have Metamask installed!");
      return;
    } else {
      console.log("Wallet exists! Happy trading!");
    }

    const accounts = await ethereum.request({ method: 'eth_accounts' });

    if (accounts.length !== 0) {
      const account = accounts[0];
      console.log("Found an authorized account: ", account);
      setCurrentAccount(account);
    } else {
      console.log("No authorized account found");
    }
  }
    
  const connectWalletHandler = async () => {
    const { ethereum } = window;

    if (!ethereum) {
      alert("Please install Metamask!");
    }

    try {
      const accounts = await ethereum.request({ method: 'eth_requestAccounts' });
      console.log("Found an account! Address: ", accounts[0]);
      setCurrentAccount(accounts[0]);
    } catch (err) {
      console.log(err)
    }
  }
    
  const connectWalletButton = () => {
    return (
      <button onClick={connectWalletHandler} className='cta-button connect-wallet-button'>
        Connect Wallet
      </button>
    )
  }

  const walletConnectedButton = () => {
    return (
      <button className='cta-button mint-nft-button'>
        Wallet Connected!
      </button>
    )
  }

  const mintNftHandler = async () => {
    try {
      const { ethereum } = window;
  
      if (ethereum) {
        const provider = new ethers.providers.Web3Provider(ethereum);
        const signer = provider.getSigner();
        const nftContract = new ethers.Contract(contractAddress, abi, signer);
  
        console.log("Initialize payment");
        let nftTxn = await nftContract.queueBuy(1, { value: ethers.utils.parseEther("0.01") });
  
        console.log("Mining... please wait");
        await nftTxn.wait();
  
        console.log(`Mined`);
      } else {
        console.log("Ethereum object does not exist");
      }
    } catch (err) {
      console.log(err);
    }
  }

  const mintNftButton = () => {
    return (
      <button onClick={mintNftHandler} className='cta-button mint-nft-button'>
        Mint NFT
      </button>
    )
  }

  return (
    <div>
        {currentAccount ? mintNftButton() : connectWalletButton()}
    </div>
  );
}
