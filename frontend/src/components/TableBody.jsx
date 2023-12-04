import contract from '../contracts/contract.json';

const ethers = require("ethers");
const contractAddress = "0xd927DD054f1B1C5a4BC56b8C71aB5e598d667713";
const abi = contract;

const TableBody = ({ tableData, columns }) => {
  const BuyItemHandler = async (uri, price) => {
    try {
      const { ethereum } = window;
  
      if (ethereum) {
        const provider = new ethers.BrowserProvider(ethereum);
        const signer = await provider.getSigner();
        const nftContract = new ethers.Contract(contractAddress, abi, signer); 
        
        console.log("Initialize payment");
        let nftTxn = await nftContract.queueBuy(uri, { value: ethers.parseEther(price.toString()) });
  
        console.log("Mining... please wait");
        await nftTxn.wait();
        
        console.log(`Mined, see transaction: https://sepolia.etherscan.io/tx/${nftTxn.hash}`);
        window.confirm(`Mined, see transaction: https://sepolia.etherscan.io/tx/${nftTxn.hash}`)
        window.location.reload();
      } else {
        console.log("Ethereum object does not exist");
      }
    } catch (err) {
      console.log(err);
    }
  }

  const SellItemHandler = async () => {
    try {
      const { ethereum } = window;
  
      if (ethereum) {
        const provider = new ethers.BrowserProvider(ethereum);
        const signer = await provider.getSigner();
        const nftContract = new ethers.Contract(contractAddress, abi, signer);
        var tokenId = prompt("Please enter your token id");
        if (tokenId != null) {
          console.log("Initialize sell");
          let nftTxn = await nftContract.queueSell(parseInt(tokenId));
    
          console.log("Burning... please wait");
          await nftTxn.wait();
    
          console.log(`Burnt, see transaction: https://sepolia.etherscan.io/tx/${nftTxn.hash}`);
          window.confirm(`Burnt, see transaction: https://sepolia.etherscan.io/tx/${nftTxn.hash}`)
          window.location.reload();
        } else {
          console.log("Invalid token id");
        }   
      } else {
        console.log("Ethereum object does not exist");
      }
    } catch (err) {
      console.log(err);
    }
  }

  return (
    <tbody>
      {tableData.map((data) => {
        return (
          <tr key={data.item}>
            {columns.map(({ accessor }) => {
              const tData = data[accessor] ? data[accessor] : "——";
              if (accessor === "buy_price") {
                return (
                  <td key={accessor}>{tData}
                  <button onClick={() => BuyItemHandler(data.uri, data.buy_price)} className="action-button buy-button">Buy</button>
                  </td>
                )
              } else if (accessor === "sell_price") {
                return (
                  <td key={accessor}>{tData}
                  <button onClick={() => SellItemHandler()} className="action-button sell-button">Sell</button>
                  </td>
                )
              } else {
                return <td key={accessor}>{tData}</td>;
              }
            })}
          </tr>
        );
      })}
    </tbody>
  );
};
  
export default TableBody;