#!/bin/sh

#storage=CHOOSE_A_NAME

#echo "Creating / Clearing $storage"
#mkdir $storage
#rm -rf $storage/*
for ip in $(aws ec2 describe-instances --query 'Reservations[*].Instances[*].PublicIpAddress' --filters 'Name=instance-state-name,Values=running' "Name=key-name,Values=jda-craft"); do
  echo "\n\n\n========="
  echo "reporting on $ip"
  #echo "Logging in to $ip"
	#mkdir $storage/$ip
  #scp -C -i YOUR_KEY -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no ubuntu@$ip:ec2_results* $storage/$ip/.
  ssh -i /Users/jda/Code/jda.pem -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no ubuntu@$ip
done

