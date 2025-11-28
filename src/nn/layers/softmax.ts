import type { ICallable } from "../interfaces/ICallable.ts";
import type {NewTensor} from "../tensor_new.ts";

export class Softmax implements ICallable {
    execute(inputs: NewTensor): NewTensor {
        return inputs.softmax();
    }
}