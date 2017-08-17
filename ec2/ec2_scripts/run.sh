BRANCH="exp/light_indep"

exec >STDOUT 2>STDERR

ssh-keyscan github.com >>/home/ubuntu/.ssh/known_hosts
su ubuntu -c "git clone git@github.com:jacobandreas/craft.git"
cd craft
git checkout $BRANCH
python main.py
git add -A
git commit -m "add experiments"
su ubuntu -c "git push"
