const nn = ml5.neuralNetwork({ task: 'classification', debug: true })

const labels = ['Victory', 'L', 'Metal'];
let testData = [];

getFromLocalStorage();

async function getFromLocalStorage () {
    for (let i = 0; i < labels.length; i++) {
        testData = [...testData, ...JSON.parse(localStorage.getItem(`testData ${i}`))];
    }
}

async function testMachine() {

    const modelDetails = {
        model: '../model/model.json',
        metadata: '../model/model_meta.json',
        weights: '../model/model.weights.bin'
    }

    await new Promise((resolve, reject) => {
        nn.load(modelDetails, (err) => {
            if (err) {
                reject(err);
            } else {
                console.log("Model Loaded!");
                resolve();
            }
        });
    });

    await tester(testData, labels);
    console.log(testData);
    await finishedTesting();
}

async function tester (data) {
    data.forEach(entry => {
        const pose = entry.pose;
        console.log(entry.label);
        nn.addData(pose, { label: entry.label });
    });
}

async function finishedTesting() {

    let correct = 0;
    let total = 0;
    for (const entry of testData) {
        const pose = entry.pose;
        const label = entry.label;

        const prediction = await nn.classify(pose);

        if (prediction[0].label === label) {
            correct++;
        }
        total++;

        console.log('prediction = ' + prediction + ' ' + correct + ' ' + total);

    }}

    console.log("Finished training!");

testMachine();
