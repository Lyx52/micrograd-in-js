import {RandomGenerator} from "./random.ts";

export class Value {
    public Data: number;
    public Grad: number;
    private _children: Value[] = [];
    private _productOf?: string;
    private _backward: (parent: Value) => void = () => {
        if (this._children.length > 1) {
            console.error("No backwards, when theres children...", this);
            throw new Error("No backwards, when theres children...");
        }
    };

    constructor(data: number, children: Value[] = [], productOf: string = null) {
        this.Data = data;
        this.Grad = 0;
        this._children = children;
        this._productOf = productOf;
    }

    public static Sum(...array: number[]|Value[]): Value {
        const result = new Value(0, [], 'sum');
        for (let value of array) {
            if (!(value instanceof Value)) {
                value = new Value(value);
            }

            result._children.push(value);
            result.Data += value.Data;

        }

        result._backward = (parent: Value): void => {
            for (const child of result._children) {
                child.Grad += parent.Grad;
            }
        }

        return result;
    }

    public static Random(): Value {
        return new Value(RandomGenerator.random(), [], null);
    }

    public clone(): Value {
        const clone = new Value(this.Data, this._children, this._productOf);
        clone._backward = this._backward;
        clone.Grad = this.Grad;

        return clone;
    }

    public zerograd() {
        this.Grad = 0;
    }

    public topological(tensors: Value[] = []): Value[] {
        if (tensors.includes(this)) {
            return tensors;
        }

        for (const child of this._children) {
            child.topological(tensors);
        }

        tensors.push(this);
        return tensors;
    }

    public backward() {
        this.Grad = 1;
        const nodes = this.topological().reverse();
        for (const node of nodes) {
            if (node._backward) {
                node._backward(node);
                continue;
            }

            console.error("Backwards not implemented on node ", node);
            throw new Error("Backwards not implemented on node ");
        }
    }

    public graph(id: number = 1, nodes: any[] = [], edges: any[] = []): [any[], any[]] {
        let nodeId = id;
        const existingNode = nodes.find(v => v.value === this);
        if (existingNode) {
            nodeId = existingNode.id;
        } else {
            nodes.push({
                id: nodeId,
                label: this.toString(),
                color: nodeId === 1 ? 'green' : 'cyan',
                value: this,
            });
        }

        if (this._productOf) {
            edges.push({
                to: nodeId,
                from: ++id,
                arrows: "to",
            });

            const operationId = id;
            nodes.push({
                id: operationId,
                label: this._productOf,
                color: 'cyan',
                value: null,
            });

            for (const child of this._children) {
                const existingNode = nodes.find(v => v.value === child);
                const childId = existingNode?.id ?? ++id;
                const [childNodes, _] = child.graph(childId, nodes, edges);

                edges.push({
                    from: childId,
                    to: operationId,
                    arrows: "to",
                });

                id = Math.max(id, ...childNodes.map(v => v.id)) + 1;
            }
        }

        return [
            nodes,
            edges
        ];
    }

    public relu() {
        const result = new Value(Math.max(0, this.Data), [this], 'relu');

        result._backward = (parent: Value) => {
            const current = parent._children[0];
            current.Grad += Math.max(0, parent.Data) * parent.Grad;
        }

        return result;
    }

    public mse(expected: number|Value): Value {
        if (expected instanceof Value) {
            return expected.clone().sub(this).pow(2)
        }

        return (new Value(expected)).sub(this).pow(2)
    }

    public tanh() {
        /**
         * x = this.Data
         * Formula: tanh(x)
         * Derivative: dL/dx = 1 / (cos^2(x))
         */
        const result = new Value(Math.tanh(this.Data), [this], 'tanh');
        result._backward = (parent: Value) => {
            const current = parent._children[0];
            const cosh_x = Math.cosh(parent.Data);
            if (Math.abs(cosh_x) < Number.EPSILON) {
                console.warn("Unknown area, cosh(x) result for tanh is close to 0?")
                current.Grad += 0;
            } else {
                current.Grad += (1 / (cosh_x * cosh_x)) *  parent.Grad;
            }
        }

        return result;
    }

    public add(other: Value|number) {
        /**
         * x = this.Data
         * y = other.Data
         * Formula: x + y
         * Derivative:
         *  -   dL/dx = L
         *  -   dL/dy = L
         */
        if (!(other instanceof Value)) {
            other = new Value(other);
        }

        const result = new Value(this.Data + other.Data, [this, other], '+');

        result._backward = (parent: Value) => {
            for (const child of parent._children) {
                child.Grad += parent.Grad;
            }
        }

        return result;
    }

    public mul(other: Value|number) {
        /**
         * x = this.Data
         * y = other.Data
         * Formula: x * y
         * Derivative:
         *  -   dL/dx = y
         *  -   dL/dy = x
         */

        if (!(other instanceof Value)) {
            other = new Value(other);
        }

        const result = new Value(this.Data * other.Data, [this, other], '*');

        result._backward = (parent: Value) => {
            const first = parent._children[0];
            const second = parent._children[1];

            first.Grad += second.Data * parent.Grad;
            second.Grad += first.Data * parent.Grad;
        }

        return result;
    }

    public pow(other: Value|number) {
        /**
         * x = this.Data
         * y = other.Data
         * Formula: x^y
         * Derivative:
         *  -   dL/dx = y * x^(y - 1)
         *  -   dL/dy = x^y * ln(y)
         */
        if (!(other instanceof Value)) {
            other = new Value(other);
        }

        const result = new Value(Math.pow(this.Data, other.Data), [this, other], '^');

        result._backward = (parent: Value) => {
            const first = parent._children[0];
            const second = parent._children[1];
            const x = first.Data;
            const z = Math.pow(x, second.Data);

            first.Grad += second.Data * Math.pow(x, (second.Data - 1)) * parent.Grad;
            second.Grad += z * Math.log(x) * parent.Grad;
        }

        return result;
    }

    public negate() {
        return this.mul(-1);
    }

    public sub(other: Value|number) {
        if (!(other instanceof Value)) {
            other = new Value(other);
        }

        return this.negate().add(other);
    }

    public div(other: Value|number) {
        /**
         * x = this.Data
         * y = other.Data
         * Formula: x/y
         * Derivative:
         *  -   dL/dx = 1 / y
         *  -   dL/dy = -x / y^2
         */
        if (!(other instanceof Value)) {
            other = new Value(other);
        }

        const result = new Value(this.Data / other.Data, [this, other], '/');

        result._backward = (parent: Value) => {
            const first = parent._children[0];
            const second = parent._children[1];
            const y = second.Data;
            const x = first.Data;
            first.Grad += (1 / y) * parent.Grad;
            second.Grad += (-x / (y * y)) * parent.Grad;
        }

        return result;
    }

    public toString() {
        return `(${this.Data.toFixed(2)}, ${this.Grad.toFixed(2)})`;
    }
}