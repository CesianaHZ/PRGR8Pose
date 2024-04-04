import kNear from "./knear.js";

const k = 3;

const machine = new kNear(k);
const nn = ml5.neuralNetwork({ task: 'classification', debug: true,
    layers: [
        {
            type: 'dense',
            units: 14,
            activation: 'relu',
        }]});

let filename;


const filenames = ['../Victory.json', '../L.json', '../Metal.json'];
const labels = ['Victory', 'L', 'Metal'];


async function fetchJSONFile(filename) {
    try {
        const response = await fetch(filename);
        if (!response.ok) {
            throw new Error(`Failed to fetch ${filename}: ${response.status} ${response.statusText}`);
        }
        return await response.json();
    } catch (error) {
        console.error(`Error fetching ${filename}: ${error}`);
        return [];
    }
}

async function trainMachine() {

    for (let i = 0; i < filenames.length; i++) {

        filename = filenames[i];
        const label = labels[i];
        const data = await fetchJSONFile(filename);

        data.sort(() => (Math.random() - 0.5));

        const train = data.slice(0, Math.floor(data.length * 0.8));
        const test = data.slice(Math.floor(data.length * 0.8) + 1);
        localStorage.setItem(`testData ${i}`, JSON.stringify(test));
        await trainer(train, label);
    }
    nn.normalizeData();

    nn.train({ epochs: 32 }, () => { finishedTraining() });
}

async function trainer (data, label) {
    data.forEach(entry => {
        const pose = entry.pose;
        //machine.learn(pose, label);\
        console.log(label);
        nn.addData(pose, { label: label });
    });
}

async function finishedTraining() {

    nn.save("model", () => console.log("model was saved!"))

    console.log("Finished training!");
}

trainMachine();
