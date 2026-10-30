// CI/CD pipeline for adfusion (declarative Jenkins pipeline, multibranch job).
//
// Every branch:  data guard -> test image -> lint -> unit tests (JUnit + coverage)
//                -> runtime image -> smoke test of the image (preprocessing and training)
// main only:     push to Docker Hub (if PUSH_IMAGE=true) -> deploy to staging
//                -> (optional, parameter DEPLOY_K8S) Kubernetes Job
//
// The Jenkins server itself is defined in ci/ (docker compose + configuration as code).
// The multibranch job scans the repository every 2 minutes, so each new commit is built.

pipeline {
    agent any

    options {
        timestamps()
        timeout(time: 45, unit: 'MINUTES')
        buildDiscarder(logRotator(numToKeepStr: '20'))
        disableConcurrentBuilds()
    }

    parameters {
        booleanParam(name: 'DEPLOY_K8S', defaultValue: false,
                     description: 'main only: also run the image as a Kubernetes Job (needs the kubeconfig credential)')
    }

    environment {
        IMAGE = 'adfusion'
    }

    stages {
        stage('Prepare') {
            steps {
                script {
                    env.SHORT_SHA = sh(script: 'git rev-parse --short=7 HEAD', returnStdout: true).trim()
                    env.VERSION = sh(script: '''python3 -c "import tomllib; print(tomllib.load(open('pyproject.toml', 'rb'))['project']['version'])"''',
                                     returnStdout: true).trim()
                    env.TEST_CONTAINER = "adfusion-test-${env.BUILD_TAG}".replaceAll('[^A-Za-z0-9_.-]', '-')
                    currentBuild.description = "${env.SHORT_SHA} (v${env.VERSION})"
                }
                sh '''
                    git log -1 --format="commit %h by %an: %s"
                    docker version --format "Docker server {{.Server.Version}}"
                '''
            }
        }

        stage('Data guard') {
            steps {
                // No imaging files, checkpoints or real ADNI subject IDs may be committed.
                sh 'python3 scripts/check_no_data.py'
            }
        }

        stage('Build test image') {
            steps {
                sh 'docker build --target test -t "$IMAGE-test:$SHORT_SHA" .'
            }
        }

        stage('Lint') {
            steps {
                sh 'docker run --rm "$IMAGE-test:$SHORT_SHA" sh -c "ruff check . && ruff format --check ."'
            }
        }

        stage('Unit tests') {
            steps {
                sh '''
                    rm -rf reports && mkdir -p reports
                    docker run --name "$TEST_CONTAINER" "$IMAGE-test:$SHORT_SHA" \
                        pytest --junitxml=/tmp/reports/junit.xml \
                               --cov=adfusion --cov-report=term \
                               --cov-report=xml:/tmp/reports/coverage.xml \
                               --cov-fail-under=85
                '''
            }
            post {
                always {
                    sh '''
                        docker cp "$TEST_CONTAINER":/tmp/reports/. reports/ || true
                        docker rm -f "$TEST_CONTAINER" >/dev/null 2>&1 || true
                    '''
                    junit allowEmptyResults: true, testResults: 'reports/junit.xml'
                    recordCoverage(tools: [[parser: 'COBERTURA', pattern: 'reports/coverage.xml']])
                    archiveArtifacts artifacts: 'reports/*.xml', allowEmptyArchive: true
                }
            }
        }

        stage('Build image') {
            steps {
                sh '''
                    docker build --target runtime \
                        --build-arg VERSION="$VERSION" --build-arg VCS_REF="$SHORT_SHA" \
                        -t "$IMAGE:$SHORT_SHA" -t "$IMAGE:latest" .
                    docker image ls "$IMAGE"
                '''
            }
        }

        stage('Smoke test image') {
            steps {
                sh '''
                    docker run --rm "$IMAGE:$SHORT_SHA" --version
                    docker run --rm "$IMAGE:$SHORT_SHA" info
                    docker run --rm --entrypoint sh "$IMAGE:$SHORT_SHA" -c "
                        set -e
                        adfusion synth --out /tmp/raw --n 4
                        adfusion preprocess --raw /tmp/raw --labels /tmp/raw/labels.csv --out /tmp/cache --shape 32
                        adfusion qc /tmp/cache/cache_manifest.csv --threshold 0.5 --strict
                        adfusion synth-cohort --out /tmp/cohort --n 24 --shape 16
                        adfusion train --manifest /tmp/cohort/cache_manifest.csv --out /tmp/results --config /app/configs/smoke.json --models pet intermediate --no-checkpoints
                        adfusion figures /tmp/results --dpi 80"
                '''
            }
        }

        stage('Push to Docker Hub') {
            when {
                allOf {
                    branch 'main'
                    environment name: 'PUSH_IMAGE', value: 'true'
                }
            }
            steps {
                withCredentials([usernamePassword(credentialsId: 'dockerhub',
                                                  usernameVariable: 'DH_USER',
                                                  passwordVariable: 'DH_TOKEN')]) {
                    sh '''
                        echo "$DH_TOKEN" | docker login -u "$DH_USER" --password-stdin
                        for t in "$SHORT_SHA" "$VERSION" latest; do
                            docker tag "$IMAGE:$SHORT_SHA" "$DH_USER/$IMAGE:$t"
                            docker push "$DH_USER/$IMAGE:$t"
                        done
                        docker logout
                    '''
                }
            }
        }

        stage('Deploy to staging') {
            when { branch 'main' }
            steps {
                sh '''
                    IMAGE_TAG="$SHORT_SHA" docker compose -f deploy/compose.staging.yml up -d --force-recreate
                    status=missing
                    for i in $(seq 1 150); do
                        status=$(docker inspect -f '{{.State.Health.Status}}' adfusion-staging 2>/dev/null || echo missing)
                        echo "staging health: $status"
                        [ "$status" = healthy ] && break
                        [ "$status" = unhealthy ] && break
                        sleep 2
                    done
                    if [ "$status" != healthy ]; then docker logs adfusion-staging; exit 1; fi
                    docker exec adfusion-staging python -c "import urllib.request; print(urllib.request.urlopen('http://localhost:8000/results/results_summary.csv').read().decode())"
                    echo "Staging is up: http://localhost:8081"
                '''
            }
        }

        stage('Deploy to Kubernetes') {
            when {
                allOf {
                    branch 'main'
                    expression { params.DEPLOY_K8S }
                }
            }
            steps {
                withCredentials([file(credentialsId: 'kubeconfig', variable: 'KUBECONFIG')]) {
                    sh '''
                        JOB="adfusion-smoke-$SHORT_SHA-$BUILD_NUMBER"
                        sed -e "s|__JOB__|$JOB|" -e "s|__IMAGE__|$IMAGE:$SHORT_SHA|" k8s/job.yaml | kubectl apply -f -
                        kubectl wait --for=condition=complete "job/$JOB" --timeout=300s
                        kubectl logs "job/$JOB"
                    '''
                }
            }
        }
    }

    post {
        always {
            sh '''
                docker rmi -f "$IMAGE-test:$SHORT_SHA" >/dev/null 2>&1 || true
                docker image prune -f >/dev/null 2>&1 || true
            '''
            cleanWs()
        }
        success {
            echo "Build ${env.SHORT_SHA} passed: image ${env.IMAGE}:${env.SHORT_SHA}"
        }
        failure {
            echo 'Build failed: see the stage log above.'
        }
    }
}
