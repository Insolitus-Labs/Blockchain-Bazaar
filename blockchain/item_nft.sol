pragma solidity ^0.8.20;


import  "@openzeppelin/contracts/token/ERC721/ERC721.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/token/ERC721/extensions/ERC721Burnable.sol";
import "@openzeppelin/contracts/token/ERC721/extensions/ERC721URIStorage.sol";

import {FunctionsClient} from "@chainlink/contracts/src/v0.8/functions/dev/v1_0_0/FunctionsClient.sol";
import {FunctionsRequest} from "@chainlink/contracts/src/v0.8/functions/dev/v1_0_0/libraries/FunctionsRequest.sol";

contract BlockChainBazaar is  ERC721URIStorage, Ownable, FunctionsClient  {
    using FunctionsRequest for FunctionsRequest.Request;

    uint256 public mint_count;
    uint64 private subscriptionId = 1662;


    bytes32 public prev_req_id;  // turn private
    bytes public s_lastResponse; // turn private
    bytes public s_lastError;
    uint32 gasLimit = 300000;

    bytes32 donID =
        0x66756e2d657468657265756d2d7365706f6c69612d3100000000000000000000;
    address router = 0xb83E47C2bC239B3bf370bc41e1459A34b41238D0;

    mapping(bytes32 => Order) order_book;

    struct Order {
      address caller_id;
      bool order_type; // Buy: 0  ||  Sell: 1
      uint256 _tokenId;
      uint256 value;
      string uri;
    }

    event RequestPrice(bytes32 indexed requestId, uint256 price);

    constructor() FunctionsClient(router) ERC721("Temp Item", "TMP") Ownable(msg.sender) {
		  mint_count = 0;
    }

    fallback() external payable {
      //getCost(msg.sender,false,0x0,msg.value);
    }
    receive() external payable {
      //getCost(msg.sender,false,0x0,msg.value);
    }

    /* */
    uint256 public price;
    function test_mint(string memory uri) public returns(uint256) {
      _mint(msg.sender,mint_count);
      _setTokenURI(mint_count,uri);
      mint_count += 1; 
      return mint_count-1;
    }


    function queueBuy(string memory uri) public payable {
      getCost(msg.sender,false,0x0,msg.value, uri);
    }

    function queueSell(uint256 _tokenId) public payable {
      require(ownerOf(_tokenId) != address(0), "Token Does Not Exist");
		
      require(msg.sender == ownerOf(_tokenId), "You are not the owner of this NFT.");

      getCost(msg.sender,true,_tokenId,msg.value,"");

    }

    string source =
        "const apiResponse = await Functions.makeHttpRequest({"
        "url: `https://min-api.cryptocompare.com/data/pricemultifull?fsyms=ETH&tsyms=USD`" /* Concat URI with in the URL to protect against forgery */
        "});"
        "if (apiResponse.error) {"
        "throw Error('Request failed');"
        "}"
        "const { data } = apiResponse;"
        "const val = parseInt(data.RAW.ETH.USD.VOLUME24HOUR);"
        "return Functions.encodeUint256(val);"; /* TODO */


    function getCost(address sender, bool order_type, uint256 token_id, uint256 value, string memory uri) public returns(bytes32) {
      
      FunctionsRequest.Request memory req;
      req.initializeRequestForInlineJavaScript(source);

      bytes32 req_id = _sendRequest(
            req.encodeCBOR(),
            subscriptionId,
            gasLimit,
            donID
        );

      prev_req_id = req_id;
      order_book[req_id] = Order(sender,order_type,token_id,value,uri);

		  return req_id;
    }


    function fulfillRequest(
        bytes32 requestId,
        bytes memory response,
        bytes memory err
    ) internal override {
        s_lastResponse = response;
        s_lastError = err;
        price = bytesToUint(response); // Temporary

        if(order_book[requestId].order_type == false){
          buy(order_book[requestId],price);
        }
        else{
          sell(order_book[requestId],price);
        }
        emit RequestPrice(requestId, price);
    }

    function buy(Order memory o, uint256 cost) public payable {
        /*if(o.value < cost){
            (bool success,) = o.caller_id.call{gas:10000,value:o.value}("");
            require(success,"Cannot Refund Eth");
            return;
        }*/
        _mint(o.caller_id,mint_count);
        _setTokenURI(mint_count,o.uri);
		    mint_count += 1;
    }

    function sell(Order memory o, uint256 cost) public payable {
		  _burn(o._tokenId);
		  (bool success,) = o.caller_id.call{gas:10000,value:cost}("");
		  require(success,"Cannot Send Eth");
    }

    function bytesToUint(bytes memory b) public pure returns (uint256){
        uint256 number;
        for(uint i=0;i<b.length;i++){
            number = number + uint8(b[i])*(2**(8*(b.length-(i+1))));
        }
        return number;
    }

    function getURI(uint256 token_id) public view returns (string memory){
        return tokenURI(token_id);
    }
}