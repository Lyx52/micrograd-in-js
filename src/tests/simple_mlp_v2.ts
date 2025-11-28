import {MLP} from "../nn/mlp/mlp.ts";
import {crossEntropyLoss, maxMarginLoss} from "../nn/utils.ts";
import {NewTensor} from "../nn/tensor_new.ts";
import {LinearModule} from "../nn/module/linear_module.ts";
import {Linear} from "../nn/layers/linear.ts";
import {RandomGenerator} from "../nn/random.ts";

export const runSimpleMLP2 = () => {
    RandomGenerator.Seed(0)
    const mlp = new LinearModule(
        new Linear(3, 4),
        new Linear(4, 4),
        new Linear(4, 1),
    );

    const xs = [
        NewTensor.from([2.0, 3.0, -1.0]),
        NewTensor.from([3.0, -1.0, 0.5]),
        NewTensor.from([0.5, 1.0, 1.0]),
        NewTensor.from([1.0, 1.0, -1.0]),
    ];
    const ys = [
        NewTensor.from([1.0]),
        NewTensor.from([1.0]),
        NewTensor.from([-1.0]),
        NewTensor.from([1.0]),
    ];

    // Train
    const learningRate = 0.01;
    const steps = 8;
    for (let i = 0; i <= steps; i++) {
        mlp.updateParameters((tensor: NewTensor) => {
            for (let j = 0; j < tensor.backing.length; j++) {
                tensor.backing[j] += -learningRate * tensor.gradients[j];
            }
        });

        const loss = maxMarginLoss(mlp, ys, xs);
        mlp.zerograd();
        loss.backward();
        console.log(`[Step ${i}/${steps}] loss: ${loss.scalar()}`)
    }
    console.log(mlp.execute(xs[2]))
}