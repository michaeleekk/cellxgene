import React from "react";
import _ from "lodash";
import Helmet from "react-helmet";
import Container from "./framework/container";
import buttonStyles from "./framework/buttons.css";
import { connect } from "react-redux";

import Categorical from "./categorical/categorical";
import Continuous from "./continuous/continuous";
import Joy from "./joy/joy";
import Graph from "./graph/graph";
import * as globals from "../globals";

import SectionHeader from "./framework/sectionHeader";

@connect((state) => {
  return {
    foo123: state
  }
})
class Home extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      expressions: null,
      vertices: null,
      coloring: null,
      metadata: null,
    };
  }
  _onURLChanged () {
    this.props.dispatch({type: 'url changed', url: document.location.href});
  };
  componentDidMount() {

    /* listen for url changes, fire one when we start the app up */
    window.addEventListener('popstate', this._onURLChanged);
    this._onURLChanged();

    /* fire an initialize action */
    const expressions = fetch(`${globals.API.prefix}${globals.API.version}expression`, {
      method: "post",
      headers: new Headers({
        'Content-Type': 'application/json'
      }),
      body: JSON.stringify({
        // "celllist": ["1001000173.G8", "1001000173.D4"],
        "genelist": ["1/2-SBSRNA4", "A1BG", "A1BG-AS1", "A1CF", "A2LD1", "A2M", "A2ML1", "A2MP1", "A4GALT"]
      })
    })
    // const expressions = fetch(`${prefix}${version}${expression}`)
      .then((res) => res.json())
      .then((data) => { this.setState({expressions: data}) })

    const initialize = fetch(`${globals.API.prefix}${globals.API.version}initialize`)
      .then((res) => res.json())
      .then((data) => { this.setState({initialize: data}) })

    const graph = fetch(`${globals.API.prefix}${globals.API.version}graph`)
      .then((res) => res.json())
      .then((data) => { this.setState({vertices: data}) })

  }

  componentWillUpdate() {
    this.props.dispatch({type: "initialize app", data: "kitty! :D"})
  }

  createExpressionsCountsMap () {

    const CHANGE_ME_MAGIC_GENE_INDEX = 5;

    const expressionsCountsMap = {};

    /* currently selected gene */
    expressionsCountsMap.geneName = this.state.expressions.data.genes[3];

    let maxExpressionValue = 0;

    /* create map of expressions for every cell */
    this.state.expressions.data.cells.map((c) => {
      /* cellname = 234 */
      expressionsCountsMap[c.cellname] = c["e"][CHANGE_ME_MAGIC_GENE_INDEX];
      /* collect the maximum value as we iterate */
      if (c["e"][CHANGE_ME_MAGIC_GENE_INDEX] > maxExpressionValue) {
        maxExpressionValue = c["e"][CHANGE_ME_MAGIC_GENE_INDEX]
      }
    })

    expressionsCountsMap.maxValue = maxExpressionValue;

    return expressionsCountsMap;
  }

  render() {
    // console.log('app:', this.props, this.state)

    return (
      <Container>
        <Helmet title="cellxgene" />
        <SectionHeader text="Gene Selection Criteria"/> 
        {false ? <Joy data={this.state.expressions && this.state.expressions.data}/> : ""}

        <Categorical/>
        <Continuous/>
        <button
          style={{marginBottom: 20}}
          className={buttonStyles.primaryButton}>
          Compute clustering using [n] cells in current metadata selection
        </button>
        <Graph
          vertices={this.state.vertices}
          expressions={this.state.expressions}
          expressionsCountsMap={this.state.expressions && this.state.expressions ? this.createExpressionsCountsMap() : null}
          />
      </Container>
    )
  }
};

export default Home;

// <Categorical title={"Sample type"} category={types}/>
// <Categorical title={"Selection"} category={selection}/>
// <Categorical title={"Location"} category={location}/>
// <Categorical title={"Sample name"} category={names}/>
