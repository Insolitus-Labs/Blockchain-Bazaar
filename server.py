from flask import Flask, abort, request, send_from_directory

import ssl
import requests

app = Flask(__name__)

host = "127.0.0.1"
port = 5000
debug = False



@app.route("/", methods=['GET','POST'])
def index():
    return "Blockchain Bazzar"


@app.route("/price", methods=['GET'])
def api_price():
    headers = request.headers

    item_name = request.args.get('item')
    damage = request.args.get('damage')
    weight = request.args.get('weight')
    
    # request AI for prices
    price = AI.get_price(damage,weight)
    
    return price


@app.route("/smart_contract_price", methods=['GET'])
def api_price():
    headers = request.headers

    eth_sent = request.args.get('eth')
    buy_or_sell = request.args.get('order_type')
    item_name = request.args.get('item')
    damage = request.args.get('damage')
    weight = request.args.get('weight')
    
    # request AI for prices
    price = AI.get_price(damage,weight)
    if(price <= eth_sent):
        AI.update_price(buy_or_sell,damage,weight)
    
    return price


if __name__ == "__main__":
#    context = SSL.Context(SSL.TLSv1_2_METHOD)
#    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
#    context.load_cert_chain('server.crt', 'server.key')
#    ssl_context = context
    ssl_context = None
    app.run(host=host, port=port, debug=debug, ssl_context=ssl_context)

