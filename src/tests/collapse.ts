import {Value} from "../nn/value.ts";
import {collapseDimensions} from "../nn/tensor.ts";

export const testCollapse = () => {
    const x = [ [ [[new Value(1)], [new Value(2)]], [[new Value(3)], [new Value(4)]] ], [ [[new Value(5)], [new Value(6)]], [[new Value(7)], [new Value(8)]] ] ]
    console.log(collapseDimensions(x, [0, 1], (zzz, i) => {
        console.log(zzz)
        return new Value(3);
    }));
}