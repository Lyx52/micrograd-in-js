import type {ICallable} from "./ICallable.ts";
import type {Value} from "./value.ts";
import {Tensor} from "./tensor.ts";
import {Layer} from "./layer.ts";
import type {IParameterized} from "./IParameterized.ts";

export class MLP implements ICallable, IParameterized {
    private layers: Layer[] = [];
    constructor(inputs: number, outputs: number[]) {
        const sizes = [inputs, ...outputs];
        for (let i = 0; i < outputs.length; i++) {
            this.layers.push(new Layer(sizes[i], sizes[i + 1]))
        }
    }

    zerograd() {
        for (const layer of this.layers) {
            layer.zerograd();
        }
    }

    parameters(): Value[] {
        return this.layers.flatMap(l => l.parameters());
    }

    execute(inputs: Tensor): Tensor {
        let layerInput = inputs;
        for (const layer of this.layers) {
            layerInput = layer.execute(layerInput);
        }

        return layerInput;
    }
}