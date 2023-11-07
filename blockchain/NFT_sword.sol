pragma solidity ^0.8.20;

import "github.com/smartcontractkit/chainlink/evm-contracts/src/v0.6/ChainlinkClient.sol";

import  "@openzeppelin/contracts/token/ERC721/ERC721.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/token/ERC721/extensions/ERC721Burnable.sol";

contract BlockChainBazaar is ERC721, ERC721Burnable, Ownable, ChainlinkClient {
	
    uint256 private mint_count;
	address ORACLE=0x6090149792dAAeE9D1D568c9f9a6F6B46AA29eFD;
	address SEPOLIA = 0x779877A7B0D9E8603169DdbD7836e478b4624789;
    bytes32 JOB= "ca98366cc7314957b8c012c72f05aeeb";
    uint256 ORACLE_PAYMENT = (1 * LINK_DIVISIBILITY) / 10;

    constructor() ERC721("Temp Item", "TMP") {
		setChainlinkToken(SEPOLIA);
		setChainlinkOracle(ORACLE);
		mint_count = 0;
    }

    function buy() public payable {

        uint256 cost = getCost();

        if(msg.value < cost){
            (bool success,) = msg.sender.call{gas:10000,value:msg.value}("");
            require(success,"Cannot Refund Eth");
            return;
        }
        _mint(msg.sender,mint_count);
		mint_count += 1;
    }

    function sell(uint256 _tokenId) public payable {
		require(_exists(_tokenId),"Token Does Not Exist");
		
        address nftOwner = ownerOf(_tokenId);
        require(msg.sender == nftOwner, "You are not the owner of this NFT.");

		uint256 cost = getCost();
		
		_burn(msg.sender, _tokenId);
		(bool success,) = msg.sender.call{gas:10000,value:cost}("");
		require(success,"Cannot Refund Eth");
    }

    function getCost() public view returns(uint256) {
		
		Chainlink.Request memory req = 
			buildChainlinkRequest(JOB, address(this), this.fulfill.selector);
		req.add(
			'get',
			'https://blockchainbazaar.com/market/price?ticker=NFT'
		);
		//req.add('path', 'price');
		return sendChainlinkRequest(req, ORACLE_PAYMENT);
    }

}