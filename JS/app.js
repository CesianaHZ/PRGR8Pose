import kNear from "./knear.js";

const k = 3;

const machine = new kNear(k);
const nn = ml5.neuralNetwork({ task: 'classification', debug: true })
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
        data.forEach(entry => {
            const pose = entry.pose;
            //machine.learn(pose, label);\
            console.log(label);
            nn.addData(pose, { label: label });
        });
    }
    nn.normalizeData();

    nn.train({ epochs: 32 }, () => { finishedTraining() });
}

async function finishedTraining() {

    nn.save("model", () => console.log("model was saved!"))

    console.log("Finished training!");
}

trainMachine();

// trainMachine().then(() => {
//     let prediction = machine.classify([0.33722376823425293,
//         0.4850902855396271,
//         -9.241504983492632e-8,
//         0.3966033160686493,
//         0.5036208033561707,
//         -0.02314070425927639,
//         0.4614262282848358,
//         0.48257189989089966,
//         -0.03280164673924446,
//         0.5118294954299927,
//         0.46688443422317505,
//         -0.03872416540980339,
//         0.5453104376792908,
//         0.46753063797950745,
//         -0.04405711591243744,
//         0.4464579224586487,
//         0.3868473768234253,
//         -0.03984971344470978,
//         0.46271267533302307,
//         0.30396130681037903,
//         -0.05933238938450813,
//         0.4706225097179413,
//         0.24648365378379822,
//         -0.07632278650999069,
//         0.47668951749801636,
//         0.19740110635757446,
//         -0.08940056711435318,
//         0.41573742032051086,
//         0.3615805506706238,
//         -0.037866607308387756,
//         0.4045313596725464,
//         0.34611937403678894,
//         -0.06029714643955231,
//         0.38408586382865906,
//         0.37270715832710266,
//         -0.07230297476053238,
//         0.37173378467559814,
//         0.3842427134513855,
//         -0.07953863590955734,
//         0.38182950019836426,
//         0.35590532422065735,
//         -0.036124687641859055,
//         0.3699290454387665,
//         0.35515931248664856,
//         -0.05525898560881615,
//         0.35652583837509155,
//         0.3907214403152466,
//         -0.05437462776899338,
//         0.35106465220451355,
//         0.4093564450740814,
//         -0.05164115130901337,
//         0.3500160574913025,
//         0.36114203929901123,
//         -0.0363122820854187,
//         0.3464353084564209,
//         0.3647242486476898,
//         -0.05086399242281914,
//         0.3385776877403259,
//         0.3972030580043793,
//         -0.045182663947343826,
//         0.3328015208244324,
//         0.41867250204086304,
//         -0.03794700652360916]);
//
//     console.log(`knn thinks this is a ${prediction}!`);
// }).catch(error => {
//     console.error('Failed to train machine:', error);
// });

// machine.learn([6, 5, 9], 'cat');
// machine.learn([12, 20, 19], 'dog');
// machine.learn([9.2, 8.1, 2],'cat');
// machine.learn([17, 15.5, 5], 'dog');
// machine.learn([9.1, 9, 1.95], 'cat');
// machine.learn([20, 20, 6.2], 'dog');
// machine.learn([9.0, 10, 2.1], 'cat');
// machine.learn([16.7, 16, 3.3], 'dog');
//
// let prediction = machine.classify([12, 18, 17]);
