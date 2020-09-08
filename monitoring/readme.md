Use the following steps to view the TRITON dashboard(s):

1. From within this repository, run `docker-compose up`
2. Navigate to [localhost:3000](http://localhost:3000) to access Grafana dashboards
```
Username: admin
Password: admin
```
3. Go to add a datasource
* Choose Prometheus 
* Use URL http://localhost:9090
* Save and test

4. Add a dashboard (+ symbol on the left)
* Click "Import"
* Chose Upload .json file and choose the `dashboard.json` from this folder

Dashboard should be visible under `Triton Inference Server` on the main page.