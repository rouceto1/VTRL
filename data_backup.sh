#!/bin/bash

while true; do
echo "Possible places:"
ls -l backups/ | awk '{print  $6, $7, $9}'

read -p "Where to back up data (cancel)" yn

case $yn in
	cancel ) break;;
	* ) echo saving;
	  mkdir backups/$yn;
	  rsync -av --progress ../VTRL/experiments/*  backups/$yn;
	  rsync -av --progress ../VTRL2/experiments/*  backups/$yn;
	  rsync -av --progress ../VTRL3/experiments/*  backups/$yn;
	  rsync -av --progress ../VTRL4/experiments/*  backups/$yn;
	  echo "--------------------One more backup try:----------------------";
    rsync -av --progress ../VTRL/experiments/*  backups/$yn;
	  rsync -av --progress ../VTRL2/experiments/*  backups/$yn;
	  rsync -av --progress ../VTRL3/experiments/*  backups/$yn;
	  rsync -av --progress ../VTRL4/experiments/*  backups/$yn;
	  break;;
esac
done

while true; do
read -p "Reset experiment (yes/no) " yn
case $yn in
    yes ) rm -rf ../VTRL/experiments/*;
      rm -rf ../VTRL2/experiments/*;
      rm -rf ../VTRL3/experiments/*;
      rm -rf ../VTRL4/experiments/*;
		exit;;
	no ) echo exiting...;
		exit;;
	* ) echo invalid response;;
esac
done
