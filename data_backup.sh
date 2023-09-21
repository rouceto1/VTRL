#!/bin/bash

while true; do
echo "Possible places:"
ls backups

read -p "Where to back up data (cancel)" yn

case $yn in
	cancel ) break;;
	* ) echo saving;
	  mkdir backups/$yn;
	  rsync -av --progress experiments/*  backups/$yn;
	  break;;
esac
done

while true; do
read -p "Reset experiment (yes/no) " yn
case $yn in
    yes ) rm -rf experiments/*;
		exit;;
	no ) echo exiting...;
		exit;;
	* ) echo invalid response;;
esac
done
