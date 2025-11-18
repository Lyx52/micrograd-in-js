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
        return new Value((Math.random() * -2) + 1, [], null);
    }

    public static Mse(expected: number|Value, result: number|Value): Value {
        if (expected instanceof Value) {
            return expected.clone().sub(result).pow(2)
        }

        return (new Value(expected)).sub(result).pow(2)
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

    public tanh() {
        const x = this.Data;
        const t = (Math.exp(2 * x) - 1) / (Math.exp(2 * x) + 1);
        const result = new Value(t, [this], 'tanh');

        result._backward = (parent: Value) => {
            const current = parent._children[0];
            current.Grad += (1 - Math.pow(t, 2)) *  parent.Grad;
        }

        return result;
    }

    public add(other: Value|number) {
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
        if (!(other instanceof Value)) {
            other = new Value(other);
        }

        const result = new Value(Math.pow(this.Data, other.Data), [this, other], '^');

        result._backward = (parent: Value) => {
            const first = parent._children[0];
            const second = parent._children[1];

            first.Grad += second.Data * Math.pow(first.Data, (second.Data - 1)) * parent.Grad;
            second.Grad += first.Data * Math.pow(second.Data, (first.Data - 1)) * parent.Grad;
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
        if (!(other instanceof Value)) {
            other = new Value(other);
        }

        return this.mul(other.pow(-1));
    }

    public toString() {
        return `(${this.Data.toFixed(2)}, ${this.Grad.toFixed(2)})`;
    }
}