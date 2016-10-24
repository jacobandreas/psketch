#!/bin/sh

for ip in $@ ; do
  echo $ip
  ssh -i jkk.pem -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no ubuntu@$ip "sudo shutdown -h now"
done

