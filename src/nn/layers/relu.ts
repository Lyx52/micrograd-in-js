import type { ICallable } from "../interfaces/ICallable.ts";
import type {NewTensor} from "../tensor_new.ts";

export class ReLu implements ICallable {
    execute(inputs: NewTensor): NewTensor {
        return inputs.relu();
    }
}