#!/bin/bash

echo "Starting Websocket Server... on port 8080"
pm2 start Server_side/Server.py --interpreter=/Users/anita/Documents/ParkingProjectFlask/.venv/bin/python --name=parking-server

echo "Start Ngrok tunnel..." 
pm2 start "ngrok http 8080 --domain=sunbird-joint-pelican.ngrok-free.app" --name ngrok-tunnel

echo "Both are running! Ready to connect at:"
echo "wss://sunbird-joint-pelican.ngrok-free.app"
echo "You can stop the server with: pm2 stop all"

