import { createApp } from 'vue';
import {createBootstrap} from "bootstrap-vue-next";
import './css/styles.css';
import App from './App.vue';

import 'bootstrap/dist/css/bootstrap.css'
import 'bootstrap-vue-next/dist/bootstrap-vue-next.css'
import {runSimpleMLP2} from "./tests/simple_mlp_v2.ts";
import {runMinst} from "./tests/minst.ts";
//
// const app = createApp(App);
// app.use(createBootstrap());
// app.mount('#app');

runMinst();