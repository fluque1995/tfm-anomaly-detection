#!/bin/bash

echo "Creating dataset folders..."
mkdir -p dataset/{train/{abnormal,normal},test}
echo "Creating features folders..."
mkdir -p {raw,processed}_{c3d,lstm}_features/{train/{abnormal,normal},test}
echo "Creating predictions folders..."
mkdir -p predictions_{c3d,lstm}
echo "Done"


