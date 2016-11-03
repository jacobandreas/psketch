#!/bin/bash

if [ "$#" -lt 4 ]; then
	echo -e "Usage:\n$0 [spot | instance] <instance type> <zone> [<price>] <script files>"
	exit 0
fi

request=$1
instance=$2
zone=$3
vpc="vpc-539e9336"
subnet="unknown"
if [ "$zone" = "us-east-1b" ]; then
  subnet="subnet-db3c31e1"
elif [ "$zone" = "us-east-1c" ]; then
  subnet="subnet-1791cd60"
elif [ "$zone" = "us-east-1d" ]; then
  subnet="subnet-af3c4df6"
elif [ "$zone" = "us-east-1e" ]; then
  subnet="subnet-f69d06dd"
fi

image="ami-3d5c002a"
key="jda"
security_group="sg-1e3dfc78"

if [ "$request" = "instance" ]; then
  for script in ${@:4} ; do
    cat template_top.sh $script template_bottom.sh > $script.complete
    userdata=$(cat $script.complete)

    aws ec2 run-instances \
      --image-id "$image" \
      --count 1 \
      --instance-type "$instance" \
      --key-name "$key" \
      --security-group-ids "$security_group" \
      --query 'Instances[0].InstanceId' \
      --user-data "$userdata" \
      --subnet-id "$subnet" \
      --associate-public-ip-address \
      --instance-initiated-shutdown-behavior terminate > $script.instance

    job_id=`cat $script.instance`
    aws ec2 create-tags --resources "$job_id" --tags "Key=Name,Value=$script"
    echo "Started $script -- $job_id with $instance $image" | tee -a job_log.txt
  done
elif [ "$request" = "spot" ]; then
  for script in ${@:5} ; do
    price=$4
    cat template_top.sh $script template_bottom.sh > $script.complete
    userdata=$(cat $script.complete | base64)

    aws ec2 request-spot-instances \
      --spot-price $price \
      --instance-count 1 \
      --type "one-time" \
      --launch-specification "{ \"KeyName\": \"$key\", \"ImageId\": \"$image\", \"InstanceType\": \"$instance\" , \"UserData\": \"$userdata\", \"Placement\": {\"AvailabilityZone\": \"$zone\"}, \"NetworkInterfaces\": [ { \"DeviceIndex\":0, \"SubnetId\":\"$subnet\", \"AssociatePublicIpAddress\": true, \"Groups\": [\"$security_group\"] } ] }" >> spot_requests.log

    echo "Started $script -- $instance $image $zone $price" | tee -a job_log.txt
  done
else
  echo "Instance type must be either  spot  or  instance"
fi

