import type { ICallable } from "./ICallable.ts";
import type {IModule} from "./IModule.ts";
import type {Tensor} from "./tensor.ts";
import type {IParameterized} from "./IParameterized.ts";
import type { Value } from "./value.ts";

export class LinearModule implements IModule, ICallable, IParameterized {
    private _layers: ICallable[] = [];

    constructor(...layers: ICallable[]) {
        this._layers = layers;
    }

    parameters(): Value[] {
        const params: Value[] = [];
        for (const layer of (this._layers as unknown[] as IParameterized[])) {
            if (layer.parameters) {
                params.push(...layer.parameters());
            }
        }

        return params;
    }

    zerograd() {
        for (const layer of (this._layers as unknown[] as IParameterized[])) {
            if (layer.zerograd) {
                layer.zerograd();
            }
        }
    }

    execute(inputs: Tensor): Tensor {
        return this.forward(inputs);
    }

    forward(input: Tensor): Tensor {
        let x = input.clone();
        for (const layer of this._layers) {
            x = layer.execute(x);
        }

        return x;
    }
}