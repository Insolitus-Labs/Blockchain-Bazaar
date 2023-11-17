pragma solidity ^0.8.20;

import "@chainlink/contracts/src/v0.8/ChainlinkClient.sol";
import "@chainlink/contracts/src/v0.8/shared/access/ConfirmedOwner.sol";

import  "@openzeppelin/contracts/token/ERC721/ERC721.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/token/ERC721/extensions/ERC721Burnable.sol";

contract BlockChainBazaar is ERC721, ERC721Burnable, Ownable, ChainlinkClient {
    using Chainlink for Chainlink.Request;

    uint256 private mint_count;
    address private immutable ORACLE  = 0x6090149792dAAeE9D1D568c9f9a6F6B46AA29eFD;
    address private immutable SEPOLIA_LINK = 0x779877A7B0D9E8603169DdbD7836e478b4624789;
    bytes32 private immutable JOB  = "ca98366cc7314957b8c012c72f05aeeb";
    uint256 private immutable ORACLE_PAYMENT = (1 * LINK_DIVISIBILITY) / 10;

    mapping(bytes32 => Order) order_book;

    struct Order {
      address caller_id;
      bool order_type; // Buy: 0  ||  Sell: 1
      uint256 _tokenId;
      uint256 value;
    }

    event RequestPrice(bytes32 indexed requestId, uint256 price);

    constructor() Ownable(msg.sender) ERC721("Temp Item", "TMP") {
		  setChainlinkToken(SEPOLIA_LINK);
		  setChainlinkOracle(ORACLE);
		  mint_count = 0;
    }

    function queueBuy() public payable {
        getCost(msg.sender,false,0x0,msg.value);
    }

    function queueSell(uint256 _tokenId) public payable {
      require(ownerOf(_tokenId) != address(0), "Token Does Not Exist");
		
      require(msg.sender == ownerOf(_tokenId), "You are not the owner of this NFT.");

      getCost(msg.sender,true,_tokenId,msg.value);

    }

    function getCost(address sender, bool order_type, uint256 token_id, uint256 value) public returns(bytes32) {
      
      Chainlink.Request memory req = 
                  buildChainlinkRequest(JOB, address(this), this.fulfill.selector);
      req.add("get", 'https://pokeapi.co/api/v2/pokemon/1');
      req.add("path", "price");
      bytes32 req_id = sendChainlinkRequest(req, ORACLE_PAYMENT);

      order_book[req_id] = Order(sender,order_type,token_id,value);

		  return req_id;
    }

    function fulfill(
        bytes32 _requestId,
        uint256 _price
    ) public recordChainlinkFulfillment(_requestId) {
        emit RequestPrice(_requestId, _price);
        Order memory o = order_book[_requestId];
        
        if(o.order_type == false){
          buy(o,_price);
        }
        else{
          sell(o,_price);
        }
    }

    function buy(Order memory o ,uint256 cost) public payable {
        if(o.value < cost){
            (bool success,) = o.caller_id.call{gas:10000,value:o.value}("");
            require(success,"Cannot Refund Eth");
            return;
        }
        _mint(o.caller_id,mint_count);
		    mint_count += 1;
    }

    function sell(Order memory o, uint256 cost) public payable {
		  _burn(o._tokenId);
		  (bool success,) = o.caller_id.call{gas:10000,value:cost}("");
		  require(success,"Cannot Send Eth");
    }

    /**
     * Allow withdraw of Link tokens from the contract
     */
    function withdrawLink() public onlyOwner {
        LinkTokenInterface link = LinkTokenInterface(chainlinkTokenAddress());
        require(
            link.transfer(msg.sender, link.balanceOf(address(this))),
            "Unable to transfer"
        );
    }

}