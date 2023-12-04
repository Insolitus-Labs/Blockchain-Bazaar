import contract from '../contracts/contract.json';

const ethers = require("ethers");
const contractAddress = "0x2fF18602C615b408d182FEE8591ad87968d1c644";
const abi = contract;

const mintNftHandler = async () => {
  try {
    const { ethereum } = window;

    if (ethereum) {
      const provider = new ethers.BrowserProvider(ethereum);
      const signer = await provider.getSigner();
      const nftContract = new ethers.Contract(contractAddress, abi, signer);

      console.log("Initialize payment");
      //let nftTxn = await nftContract.queueBuy(1, { value: ethers.utils.parseEther("0.01") });

      console.log("Mining... please wait");
      // await nftTxn.wait();

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

const TableBody = ({ tableData, columns }) => {
    return (
      <tbody>
        {tableData.map((data) => {
          return (
            <tr key={data.item}>
              {columns.map(({ accessor }) => {
                const tData = data[accessor] ? data[accessor] : "——";
                if (accessor === "buy_price") {
                  return (
                    <>
                      <td key={accessor}>{tData}
                      <button className="action-button buy-button">Buy</button>
                      </td>
                    </>
                  )
                } else if (accessor === "sell_price") {
                  return (
                    <>
                      <td key={accessor}>{tData}
                      <button className="action-button sell-button">Sell</button>
                      </td>
                    </>
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