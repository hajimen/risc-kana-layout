import Phaser from 'phaser';
import Cookies from 'js-cookie';
import { wit_text } from './wit-text';
import { Md5 } from 'ts-md5';

class BasicButton {
  constructor(
    public image: Phaser.GameObjects.Image,
    public listener: any,
    public texts: Phaser.GameObjects.Text[],
    public enabled: boolean){}
}

class ButtonRegion {
  buttons: BasicButton[] = [];

  constructor(protected scene: Phaser.Scene){
    const si = this.scene.input;
    si.on('gameobjectover', (_: any, bi: Phaser.GameObjects.Image) => {
      this.setButtonFrame(bi, 0);
    });
    si.on('gameobjectout', (_: any, bi: Phaser.GameObjects.Image) => {
      this.setButtonFrame(bi, 1);
    });
    si.on('gameobjectdown', (_: any, bi: Phaser.GameObjects.Image) => {
      this.setButtonFrame(bi, 2);
    });
    si.on('gameobjectup', (_: any, bi: Phaser.GameObjects.Image) => {
      this.setButtonFrame(bi, 0);
      const bb = this.buttons[Number(bi.getData('index'))];
      if (bb !== undefined && bb.image === bi) {
        bb.listener(bb);
      }
    });
  }

  create() {
    this.buttons.splice(0);
  }

  setButtonFrame (bi: Phaser.GameObjects.Image, frame: number) {
    bi.frame = bi.scene.textures.getFrame('button', frame);
  }

  enableRegion() {
    for (const bb of this.buttons) {
      if (!bb.enabled) {
        continue;
      }
      bb.image.setAlpha(1);
      for (const t of bb.texts) {
        t.setAlpha(1);
      }
    }
  }

  fadeInRegion() {
    let objs: Phaser.GameObjects.GameObject[] = [];
    for (const bb of this.buttons) {
      if (!bb.enabled) {
        continue;
      }
      objs.push(bb.image);
      for (const t of bb.texts) {
        objs.push(t);
      }
    }

    this.scene.tweens.add({
      targets: objs,
      alpha: 1,
      duration: 500,
    });
  }

  disableRegion() {
    for (const bb of this.buttons) {
      bb.image.setAlpha(0);
    }
  }
}

class StartButtonRegion extends ButtonRegion {
  constructor(protected scene: Phaser.Scene){
    super(scene);
  }

  addButton(text: string, listener: any, x: number, y: number) {
    const buttonImage = this.scene.add.image(x, y, 'button', 1).setOrigin(0.5);
    buttonImage.setData('index', this.buttons.length);
    buttonImage.setAlpha(0);
    buttonImage.setScale(1.5);
    buttonImage.setInteractive({ useHandCursor: true  } );

    const ti = this.scene.add.text(x, y - 3, text, { fontSize: '18px', color: '#000', fontFamily: 'monomaniac-one' }).setOrigin(0.5);
    ti.setAlpha(0);

    const bb = new BasicButton(buttonImage, listener, [ti], true);
    this.buttons.push(bb);
    return bb;
  }
}


const subtleStyle = {
  fontSize: '14px', color: '#808080', fontFamily: 'monomaniac-one'
};


class IntraSceneButtonRegion extends ButtonRegion {
  xPos: number;
  versionText?: Phaser.GameObjects.Text;
  creditText?: Phaser.GameObjects.Text;
  static sceneManager?: Phaser.Scenes.SceneManager;
  static scaleManager?: Phaser.Scale.ScaleManager;
  static instances: {[name: string]: IntraSceneButtonRegion} = {};

  constructor(protected scene: Phaser.Scene){
    super(scene);

    IntraSceneButtonRegion.instances[scene.scene.key] = this;
    this.xPos = Number(this.scene.game.config.width) - 10;
  }

  create() {
    super.create();
    const h = Number(this.scene.game.config.height);
    this.creditText = this.scene.add.text(this.xPos + 3, h - 3, 'CC BY-SA DecentKeyboards', subtleStyle).setOrigin(1);
    this.versionText = this.scene.add.text(3, h - 3, 'Version 20241208', subtleStyle).setOrigin(0, 1);

    if (IntraSceneButtonRegion.sceneManager === undefined) {
      IntraSceneButtonRegion.sceneManager = this.scene.game.scene;
      IntraSceneButtonRegion.scaleManager = this.scene.scale;
      this.scene.scale.on(Phaser.Scale.Events.ENTER_FULLSCREEN, (_: any) => {
        IntraSceneButtonRegion.updateFullscreen();
      });
      this.scene.scale.on(Phaser.Scale.Events.LEAVE_FULLSCREEN, (_: any) => {
        IntraSceneButtonRegion.updateFullscreen();
      });
    }

    if (this.scene.game.device.fullscreen.available) {
      const lt = this.scene.scale.isFullscreen ? 'を解除' : 'にする';
      this.addButton(['全画面', lt], (_: any) => {
        this.toggleFullscreen();
      });
    }
  }

  static updateFullscreen() {
    for (const key in IntraSceneButtonRegion.instances) {
      const sc = IntraSceneButtonRegion.sceneManager!;
      if (sc.isActive(key) || sc.isSleeping(key)) {
        const bb = IntraSceneButtonRegion.instances[key].buttons[0];
        if (IntraSceneButtonRegion.scaleManager!.isFullscreen) {
          bb.texts[0].text = '全画面';
          bb.texts[1].text = 'を解除';
        } else {
          bb.texts[0].text = '全画面';
          bb.texts[1].text = 'にする';
        }
      }
    }
  }

  toggleFullscreen () {
    if (this.scene.scale.isFullscreen) {
      this.scene.scale.stopFullscreen();
    } else {
      this.scene.scale.startFullscreen();
    }
  }

  addButton(texts: string[], listener: any, enabled=true) {
    let maxWidth = 0;
    for (const t of texts) {
      const ti = this.scene.add.text(0, 0, t, { fontSize: '14px', color: '#000', fontFamily: 'noto-sans-jp' });
      if (ti.displayWidth > maxWidth) {
        maxWidth = ti.displayWidth;
      }
      ti.destroy();
    }

    const buttonImage = this.scene.add.image(this.xPos, Number(this.scene.game.config.height) - 20, 'button', 1).setOrigin(1);
    buttonImage.setData('index', this.buttons.length);
    buttonImage.setAlpha(0);
    const xScale = (maxWidth < 80) ? 1.5 : ((maxWidth + 10) / 80) * 1.5;
    buttonImage.setScale(xScale, 1.5);
    const bc = buttonImage.getCenter();

    let i = 0;
    let textImages = [];
    for (const t of texts) {
      const offset = ((i == 0) ? -10 : 7) + (texts.length == 1 ? 8 : 0);
      const ti = this.scene.add.text(bc.x, bc.y + offset, t, { fontSize: '14px', color: '#000', fontFamily: 'noto-sans-jp' }).setOrigin(0.5);
      ti.setAlpha(0);
      textImages.push(ti);
      i ++;
    }

    if (enabled) {
      buttonImage.setInteractive({ useHandCursor: true  } );
    } else {
      buttonImage.setAlpha(0);
      for (const ti of textImages) {
        ti.setAlpha(0);
      }
    }

    const bb = new BasicButton(buttonImage, listener, textImages, enabled);
    this.buttons.push(bb);
    const bbl = buttonImage.getBottomLeft();
    this.xPos = bbl.x - 5;
    return bb;
  }

  fadeInButton(bb: BasicButton) {
    bb.enabled = true;
    bb.image.setInteractive({ useHandCursor: true  } );
    let objs: Phaser.GameObjects.GameObject[] = [this.creditText!];
    objs.push(bb.image);
    for (const t of bb.texts) {
      objs.push(t);
    }

    this.scene.tweens.add({
      targets: objs,
      alpha: 1,
      duration: 100,
    });
  }

  disableButton(bb: BasicButton) {
    bb.enabled = false;
    bb.image.removeInteractive();
    bb.image.setAlpha(0);
    for (const t of bb.texts) {
      t.setAlpha(0);
    }
  }
}

class WhatIsThisScene extends Phaser.Scene {
  pages: Phaser.GameObjects.Text[][] = [];
  ibr?: IntraSceneButtonRegion;
  nextPage: number = 0;

  constructor(){
    super({ key: 'WhatIsThisScene', active: false });
  }

  preload() {
    this.ibr = new IntraSceneButtonRegion(this);
		this.load.image('wit-keyboard', 'assets/wit-keyboard.png');
  }

  create(_arg: any){
    this.pages.splice(0);
    this.nextPage = 0;
    this.add.image(400, 300, 'wit-keyboard');
    this.add.rectangle(0, 0, Number(this.game.config.width), Number(this.game.config.height), 0x000000, 0.6).setOrigin(0);

    let i = 0;
    for (const pt of wit_text) {
      const h: string = <string>pt[0];
      const ps: string[] = <string[]>pt[1];
      const hi = this.add.text(400, 60, h, { fontSize: '32px', color: '#ffffff', fontFamily: 'noto-sans-jp' }).setOrigin(0.5);
      hi.setAlpha(0);
      let page = [];
      page.push(hi);
      let ii = 0;
      for (const p of ps) {
        const pi = this.add.text(400, 180 + 40 * ii, p, { fontSize: '20px', color: '#ffffff', fontFamily: 'noto-sans-jp' }).setOrigin(0.5);
        pi.setAlpha(0);
        page.push(pi);
        ii ++;
      }
      this.pages.push(page);
      i ++;
    }

    this.sound.stopAll();
    this.sound.playAudioSprite('audio', 'start');

    const np = (bb: BasicButton) => {
      if (this.nextPage == 0) {
        this.scene.transition({ target: 'StartScene', duration: 100 });
      }
      this.showPage(this.nextPage);
      if (this.nextPage == 0) {
        bb.texts[0].text = '最初に戻る';
      }
    };

    this.ibr!.create();
    const nb = this.ibr!.addButton(['　次に進む　'], np);

    this.input.keyboard!.on('keydown', (e: KeyboardEvent) => {
      if (e.code !== null) {
        if (e.code == 'Space') {
          np(nb);
        }
      }
    });

    this.ibr!.addButton(['RISCかな配列の', 'Wikiホームを開く'], (_bb: BasicButton) => {
      window.open('https://github.com/hajimen/risc-kana-layout/wiki');
    });
    this.ibr!.addButton(['DecentKeyboardsの', 'ネットショップを開く'], (_bb: BasicButton) => {
      window.open('https://decentkeyboards.booth.pm/');
    });
    this.ibr!.fadeInRegion();
    this.showPage(0);
  }

  showPage(i: number) {
    let ii = 0;
    let toShowObjs: Phaser.GameObjects.GameObject[] = [];
    let toHideObjs: Phaser.GameObjects.GameObject[] = [];
    for (const page of this.pages) {
      const show = (i == ii);
      for (const t of page) {
        if (t.alpha > 0.5 != show) {
          if (show) {
            toShowObjs.push(t);
          } else {
            toHideObjs.push(t);
          }
        }
      }
      ii ++;
    }
    this.tweens.add({
      targets: toShowObjs,
      alpha: 1,
      duration: 100,
    });
    this.tweens.add({
      targets: toHideObjs,
      alpha: 0,
      duration: 100,
    });
    this.nextPage = (i + 1 == wit_text.length) ? 0 : i + 1;
  }
}

class StartScene extends Phaser.Scene {
  ibr?: IntraSceneButtonRegion;
  sbr?: StartButtonRegion;
  stage?: Stage
  levelStars: Phaser.GameObjects.Text[] = [];

  constructor(){
    super({ key: 'StartScene', active: true });
  }

  preload() {
    this.load.spritesheet('button', 'assets/button.png', { frameWidth: 65, frameHeight: 28 });
		this.load.image('start', 'assets/start.jpg');
    this.load.audioSprite('audio', 'assets/audio-sprite.json');
    this.sbr = new StartButtonRegion(this);
    this.ibr = new IntraSceneButtonRegion(this);
    this.stage = new Stage(this);
  }

  addLevelButton(i: number, level: string, levelListener: any) {
    this.sbr!.addButton(level, levelListener, 700, 250 + i * 50);
    const cb = Cookies.get(level);
    if (cb !== undefined) {
      const s = '☆'.repeat(Number(cb) + 1);
      const star = this.add.text(750, 250 + i * 50, s, { fontSize: '15px', color: '#ffffff', fontFamily: 'noto-sans-jp' }).setOrigin(0, 0.5).setAlpha(0);
      this.levelStars.push(star);
    }
  }

  create(){
    const bg = this.add.image(400, 300, 'start');
    bg.setAlpha(0);
    const levelListener = (bb: BasicButton) => {
      this.scene.transition({
        target: 'StageScene',
        data: {
          level: bb.texts[0].text
        },
        duration: 1000
      });
    };
    this.addLevelButton(0, 'Basic', levelListener);
    this.addLevelButton(1, 'Advanced', levelListener);
    this.addLevelButton(2, 'Practical', levelListener);
    this.addLevelButton(3, 'Maniac', levelListener);
    this.sbr!.addButton('ナニコレ？', (_bb: BasicButton) => {
      this.scene.transition({
        target: 'WhatIsThisScene',
        duration: 100
      });
    }, 700, 450);
    // this.sbr!.addButton('Score', (_bb: BasicButton) => {
    //   this.stage!.setLevel('Basic');
    //   for (let i = 0; i < 49; i ++) {
    //     this.stage!.nextStage(2);
    //   }
    //   this.scene.transition({
    //     target: 'StageScoreScene',
    //     duration: 100,
    //     data: {
    //       stage: this.stage,
    //       time: 2000,
    //       nType: 20,
    //       missCount: 0
    //     }
    //   });
    // }, 700, 500);

    this.tweens.add({
      targets: bg,
      alpha: 1,
      duration: 200,
    });
    this.time.delayedCall(500, () => {
      this.sbr!.fadeInRegion();
      this.ibr!.fadeInRegion();
      this.add.tween({
        targets: this.levelStars,
        alpha: 1,
        duration: 300,
      });
    });
    this.ibr!.create();
    this.sound.stopAll();
  }
}

class Stage {
  level: string = '';
  current = 0;
  scene: Phaser.Scene;
  loadedImages: string[] = [];
  scoreRecord: number[] = [];
  stageData: any;
  levelData: any;
  layout: any;
  reverseLayout: { [latin: string]: string } = {};

  constructor(scene: Phaser.Scene) {
    // should be called from preload
    scene.load.json('stageData', 'assets/stage-data.json');
    scene.load.json('layout', 'assets/layout.json');
    this.scene = scene;
  }

  setLevel(level: string) {
    // should be called from create
    this.level = level;
    this.current = 0;
    this.scoreRecord = [];
    this.stageData = this.scene.cache.json.get("stageData");
    this.levelData = this.stageData[level];
    this.layout = this.scene.cache.json.get("layout");
    for (const kana in this.layout) {
      this.reverseLayout[this.layout[kana]] = kana;
    }
  }

  nextStage(score: number) {
    this.scoreRecord.push(score);
    this.current ++;
    return this.current < this.levelData.length;
  }

  getNStage() {
    return this.levelData.length;
  }

  nRun() {
    let n = 0;
    for (let i = this.scoreRecord.length - 1; i >= 0; i--) {
      if (this.scoreRecord[i] < 2) {
        break;
      }
      n ++;
    }
    return n;
  }

  tally() {
    let scoreN = [0, 0, 0, 0];
    let maxRun = 0;
    let running = false;
    let nRun = 0;
    for (let i of this.scoreRecord) {
      scoreN[i] ++;
      if (i == 2) {
        if (!running) {
          nRun = 0;
          running = true;
        }
        nRun ++;
        if (nRun > maxRun) {
          maxRun = nRun;
        }
      } else {
        running = false;
      }
    }
    scoreN[3] = maxRun;
    return scoreN;
  }

  getText(): string[] {
    return this.levelData[this.current][0];
  }

  getKana(): string[][] {
    return this.levelData[this.current][1];
  }

  getCost(): number {
    return this.levelData[this.current][2];
  }

  getLatin(): string[][] {
    let kana = this.getKana();
    let latin = [];
    for (let line of kana) {
      let ll = [];
      for (let k of line) {
        ll.push(this.layout[k]);
      }
      latin.push(ll);
    }
    return latin;
  }

  loadIllust() {
    for (const st of this.levelData) {
      const n = Md5.hashStr(st[0].join(''));
      this.scene.load.image(n, 'assets/illust/' + n + '.jpg');
    }
    this.scene.load.start();
  }

  getIllust(): string {
    return Md5.hashStr(this.getText().join(''));
  }

  getLastHashStr(): string {
    return Md5.hashStr(this.levelData[this.current - 1][0].join(''));
  }
}

const oofC = '#b0b0b0';
const jst = { fontSize: '30px', color: '#ffffff', fontFamily: 'noto-sans-jp' };
const kst = { fontSize: '30px', color: oofC, fontFamily: 'noto-sans-jp' };
const lst = { fontSize: '20px', color: '#ffffff', fontFamily: 'source-code' };
const wrongC = '#EF454A';
const rightC = '#ffffff';
const hintC = '#AFDFE4';

class StageScene extends Phaser.Scene {
  stage?: Stage;
  ibr?: IntraSceneButtonRegion;
  background?: Phaser.GameObjects.Rectangle;
  stageText?: Phaser.GameObjects.Text;
  readyText?: Phaser.GameObjects.Text;
  leftTexts: Phaser.GameObjects.Text[] = [];
  wrongKanaText?: Phaser.GameObjects.Text;
  wrongLatinText?: Phaser.GameObjects.Text;
  illustName?: string;
  timeStart?: number;
  textAll: Phaser.GameObjects.Text[][][] = [];
  rectAll: Phaser.GameObjects.Rectangle[][] = [];
  illust?: Phaser.GameObjects.Image;
  missCount = 0;
  currentLine = 0;
  currentMora = 0;
  currentLatin = 0;
  running = false;
  lastKanaText?: Phaser.GameObjects.Text;
  lastLatinTexts: Phaser.GameObjects.Text[] = [];
  onceEventRegistered = false;
  stageCleared = false;
  nextButton?: BasicButton;
  duration = 0;

  constructor(){
    super({ key: 'StageScene', active: false });
  }

  init(): void {
    if (!this.onceEventRegistered) {
      this.events.on(Phaser.Scenes.Events.WAKE, () => {
        this.sound.stopAll();
        this.resetFixture();
      });
      this.onceEventRegistered = true;
    }
  }

  preload() {
    this.stage = new Stage(this);
    this.ibr = new IntraSceneButtonRegion(this);
  }

  resetFixture() {
    this.ibr!.creditText!.text = 'CC BY-SA DecentKeyboards';
    this.missCount = 0;
    this.currentLine = 0;
    this.currentMora = 0;
    this.currentLatin = 0;
    this.running = false;
    this.textAll = [];
    this.rectAll = [];
    this.lastKanaText = undefined;
    this.lastLatinTexts = [];
    this.stageCleared = false;
    this.ibr!.disableButton(this.nextButton!);

    for (const o of [this.background, this.stageText, this.readyText]) {
      o!.setAlpha(0);
    }
    for (const o of this.leftTexts) {
      o.setAlpha(0);
    }

    this.tweens.add({
      targets: [this.background, this.stageText],
      alpha: 1,
      duration: 200,
    });

    this.illustName = this.stage!.getIllust();

    this.time.delayedCall(500, () => {
      this.tweens.add({
        targets: this.readyText,
        alpha: 1,
        duration: 200,
      });
    });

    this.time.delayedCall(2000, () => {
      this.readyText!.setAlpha(0);
      this.stageText!.setAlpha(0);
      this.runGame();
    });

    this.stageText!.text = 'Level ' + this.stage!.level + ': Stage ' + String(this.stage!.current + 1) + ' of ' + String(this.stage!.getNStage());
    this.ibr!.disableRegion();
  }

  setHintToText(objs: Phaser.GameObjects.Text[]){
    for (const t of objs) {
      t.setColor(hintC);
      t.setAlpha(1);
    }
  };

  focusCurrentKana() {
    if (this.lastKanaText !== undefined) {
      this.lastKanaText!.setColor(oofC);
      for (const t of this.lastLatinTexts) {
        t.setColor(oofC);
      }
    }
    const kanaText = this.textAll[this.currentLine][this.currentMora][0];
    kanaText.setColor('#ffffff');
    this.add.tweenchain({
      targets: kanaText,
      tweens: [{
          scale: 1,
          duration: 0,
        }, {
          scale: 2,
          repeat: 1,
          yoyo: true,
          duration: 100,
        }, {
          scale: 1,
          duration: 0,
        }
      ]
    });
    this.lastKanaText = kanaText;
    const latinTexts = this.getCurrentBox().slice(1);
    for (let i = 0; i < this.currentLatin; i ++) {
      latinTexts[i].setColor('#ffffff');
    }
    this.lastLatinTexts = latinTexts;
  }

  getCurrentBox() {
    return this.textAll[this.currentLine][this.currentMora];
  }

  updateWrong(append_c?: string, backspace=false) {
    const findMora = (s: string) => {
      for (const length of [1, 2, 3]) {
        if (s.length < length) {
          break;
        }
        const ss = s.substring(0, length);
        if (ss in this.stage!.reverseLayout) {
          return this.stage!.reverseLayout[ss];
        }
      }
      return '';
    };

    this.rectAll[this.currentLine][this.currentMora].fillColor = 0x72584e;
    const wkt = this.wrongKanaText!;
    const wlt = this.wrongLatinText!;
    const cb = this.getCurrentBox();
    let newStr = '';
    for (const t of cb.slice(1, this.currentLatin + 1)) {
      newStr += t.text;
    }

    if (backspace) {
      const ss = wlt.text.substring(0, wlt.text.length - 1);
      newStr += ss;
      wkt.text = findMora(newStr);
      wlt.text = ss;
      if (wlt.text.length == 0) {
        wkt.setAlpha(0);
        cb[0].setAlpha(1);
        wlt.setAlpha(0);
        this.setHintToText(cb.slice(this.currentLatin + 1));
      }
    } else {
      newStr += wlt.text;
      newStr += append_c!;
      wkt.text = findMora(newStr);
      if (wlt.text.length == 0) {
        this.missCount ++;
        const kt = cb[0];
        kt.setAlpha(0);
        const kxy = kt.getCenter();
        wkt.setX(kxy.x);
        wkt.setY(kxy.y);
        wkt.setAlpha(1);
        const lxy = cb[this.currentLatin + 1].getTopLeft();
        wlt.setX(lxy.x);
        wlt.setY(lxy.y);
        wlt.setAlpha(1);
        wlt.text = append_c!;
        this.sound.playAudioSprite('audio', 'buzzer');
        for (const t of cb.slice(this.currentLatin + 1)) {
          t.setAlpha(0);
        }
      } else {
        this.missCount ++;
        wlt.text += append_c!;
          this.sound.playAudioSprite('audio', 'buzzer');
      }
    }
  }

  create(data: any){
    this.stage!.setLevel(data.level);
    this.stage!.loadIllust();
    if (this.load.isReady()) {
      this.createNext();
    } else {
      this.load.on('complete', () => {
        this.createNext();
      });
    }
  }

  createNext() {
    this.background = this.add.rectangle(0, 0, Number(this.game.config.width), Number(this.game.config.height), 0x000000).setOrigin(0);
    this.stageText = this.add.text(400, 200, '', { fontSize: '40px', color: '#f8f8f8', fontFamily: 'monomaniac-one' }).setOrigin(0.5).setAlpha(0);
    this.readyText = this.add.text(400, 300, 'Ready', { fontSize: '80px', color: '#f8f8f8', fontFamily: 'monomaniac-one' }).setOrigin(0.5).setAlpha(0);
    this.wrongKanaText = this.add.text(0, 0, '', kst).setColor(wrongC).setOrigin(0.5).setPadding(4).setAlpha(0);
    this.wrongKanaText.depth = 10;
    this.wrongLatinText = this.add.text(0, 0, '', lst).setColor(wrongC).setOrigin(0).setAlpha(0);
    this.wrongLatinText.depth = 10;
    this.leftTexts = [];
    for (let i = 0; i < 5; i ++) {
      this.leftTexts.push(this.add.text(200, 60 + 50 * i, '', jst).setOrigin(0.5));
    }

    this.sound.stopAll();
    this.sound.playAudioSprite('audio', 'start');

    this.input.keyboard?.on('keydown', (ke: KeyboardEvent) => {
      if (ke.code === null || !this.running) {
        return;
      }
      if (this.stageCleared) {
        if (ke.code.startsWith('Space')) {
          this.gotoNext();
        }
        return;
      }
      // if (ke.code.startsWith('Space')) {
      //   this.duration = this.time.now - this.timeStart!;
      //   this.stageCleared = true;
      //   this.ibr!.fadeInButton(this.nextButton!);
      //   return;
      // }
      if (ke.code.startsWith('Key') || ke.code == 'Comma') {
        let c = (ke.code == 'Comma' ? ',' : ke.code[3].toLowerCase());
        if (this.wrongLatinText!.text.length > 0) {
          this.updateWrong(c);
          return;
        }
        let t = this.getCurrentBox()[this.currentLatin + 1];
        if (t.text == c) {
          t.setColor(rightC);
          t.setAlpha(1);
          this.currentLatin ++;
          if (this.getCurrentBox().length == this.currentLatin + 1) {
            this.currentLatin = 0;
            this.currentMora ++;
            if (this.textAll[this.currentLine].length == this.currentMora) {
              this.currentMora = 0;
              this.currentLine ++;
              this.sound.playAudioSprite('audio', 'bell');
              if (this.textAll.length == this.currentLine) {
                this.duration = this.time.now - this.timeStart!;
                this.lastKanaText!.setColor(oofC);
                for (const t of this.lastLatinTexts) {
                  t.setColor(oofC);
                }
                this.stageCleared = true;
                this.ibr!.fadeInButton(this.nextButton!);
                return;
              }
            }
            this.focusCurrentKana();
          }
        } else {
          this.updateWrong(c);
        }
      } else if (ke.code == 'Backspace') {
        if (this.wrongLatinText!.text.length == 0) {
          if (this.currentLatin > 0) {
            this.currentLatin --;
          } else if (this.currentMora > 0) {
            for (const t of this.getCurrentBox().slice(1)) {
              t.setAlpha(0);
            }
            this.currentMora --;
            this.currentLatin = this.getCurrentBox().length - 2;
          } else if (this.currentLine > 0) {
            for (const t of this.getCurrentBox().slice(1)) {
              t.setAlpha(0);
            }
            this.currentLine --;
            this.currentMora = this.textAll[this.currentLine].length - 1;
            this.currentLatin = this.getCurrentBox().length - 2;
          } else {
            return;
          }
          this.focusCurrentKana();
        } else {
          this.updateWrong(undefined, true);
        }
      }
    });
    this.ibr!.create();
    this.nextButton = this.ibr!.addButton(['スペースキーで', '次に進む'], (_bb: BasicButton) => {
      this.gotoNext();
    });
    this.resetFixture();
  }

  gotoNext() {
    this.running = false;
    let nType = 0;
    for (const textLine of this.textAll) {
      for (const t of textLine) {
        nType += t.length - 1;
      }
    }
    for (const rectLine of this.rectAll) {
      for (const r of rectLine) {
        r.destroy();
      }
    }
    for (const textLine of this.textAll) {
      for (const ts of textLine) {
        for (const t of ts) {
          t.destroy();
        }
      }
    }
    this.rectAll.splice(0);
    this.textAll.splice(0);
    this.illust!.setAlpha(0);
    this.scene.transition({
      target: 'StageScoreScene',
      duration: 300,
      data: {
        time: this.duration,
        missCount: this.missCount,
        nType: nType,
        stage: this.stage
      },
      sleep: true
    });
  };

  runGame() {
    this.ibr!.creditText!.text = 'CC BY-SA Wikipedia and DecentKeyboards';
    this.ibr!.enableRegion();
    if (this.illust !== undefined) {
      this.illust!.destroy();
    }
    this.illust = this.add.image(200, 300, this.illustName!).setScale(0.5).setOrigin(0.5, 0);
    this.timeStart = this.time.now;
    let i = 0;
    for (let s of this.stage!.getText()) {
      this.leftTexts[i].text = s;
      this.leftTexts[i].setAlpha(1);
      i ++;
    }
    i = 0;
    let widthAll: number[][] = [];
    for (let kl of this.stage!.getKana()) {
      let ll = this.stage!.getLatin()[i];
      let textLine: Phaser.GameObjects.Text[][] = [];
      let widthLine = [];
      let ii = 0;
      for (let k of kl) {
        let l = ll[ii];
        const kt = this.add.text(0, 0, k, kst).setOrigin(0.5).setPadding(4);
        kt.depth = 10;
        let lw = 0;
        let t = [kt];
        for (let fl of l) {
          const flt = this.add.text(0, 0, fl, lst).setOrigin(0).setAlpha(0);
          flt.depth = 10;
          lw += flt.displayWidth;
          t.push(flt);
        }
        textLine.push(t);
        widthLine.push(Math.max(kt.displayWidth, lw))
        ii ++;
      }
      this.textAll.push(textLine);
      widthAll.push(widthLine);
      i ++;
    }
    i = 0;
    for (let widthLine of widthAll) {
      let yTop = 32 + 80 * i;
      let lineWidth = widthLine.reduce((sum, current) => sum + current, 0);
      let xPos = 600 - lineWidth / 2;
      if (lineWidth > 400) {
        xPos -= (lineWidth - 400) / 2;
      }
      let ii = 0;
      let rectLine = [];
      for (let w of widthLine) {
        let t = this.textAll[i][ii];
        const c = ((ii + i) % 2) == 0 ? 0x343022 : 0x404756;
        const r =this.add.rectangle(xPos, yTop, w, 76, c, 1).setOrigin(0);
        r.depth = 0;
        rectLine.push(r);
        const xc = xPos + w / 2;
        xPos += w;
        const kt = t[0];
        kt.setX(xc);
        kt.setY(yTop + 20);
        let lw = 0;
        for (let iii = 1; iii < t.length; iii ++) {
          lw += t[iii].displayWidth;
        }
        let ox = 0;
        for (let iii = 1; iii < t.length; iii ++) {
          const fl = t[iii];
          fl.setX(xc - lw / 2 + ox);
          ox += fl.displayWidth;
          fl.setY(yTop + 45);
        }
        ii ++;
      }
      this.rectAll.push(rectLine);
      i ++;
    }


    this.focusCurrentKana();

    this.running = true;
  }
}

class ScoreSceneBase extends Phaser.Scene {
  ibr?: IntraSceneButtonRegion

  color_i = 0;
  results: Phaser.GameObjects.Text[] = [];
  hsv = Phaser.Display.Color.HSVColorWheel();

  constructor(key: string){
    super({ key: key, active: false });
  }

  preload() {
    this.ibr = new IntraSceneButtonRegion(this);
    for (let i = 0; i < 3; i ++) {
      this.load.image('score' + String(i), 'assets/score' + String(i) + '.png');
    }
  }

  update () {
    const top = this.hsv[this.color_i].color;
    const bottom = this.hsv[359 - this.color_i].color;

    for (let i = 0; i < this.results.length; i ++) {
      if (i % 2 == 0) {
        this.results[i].setTint(top, top, bottom, bottom);
      } else {
        this.results[i].setTint(top, bottom, top, bottom);
      }
    }

    this.color_i++;

    if (this.color_i === 360) {
        this.color_i = 0;
    }
  }
}

class StageScoreScene extends ScoreSceneBase {
  constructor(){
    super('StageScoreScene');
  }

  create(data: any) {
    const background = this.add.rectangle(0, 0, Number(this.game.config.width), Number(this.game.config.height), 0x000000).setOrigin(0);
    background.setAlpha(0);

    const stage: Stage = data.stage;
    const mc = (data.missCount == 0) ? 'Zero!' : String(data.missCount);
    const t: number = data.time / 1000;
    const titleText = this.add.text(400, 50, 'Stage Score', { fontSize: '80px', color: '#f8f8f8', fontFamily: 'monomaniac-one' }).setOrigin(0.5).setAlpha(0);
    const labelStyle = { fontSize: '40px', color: '#d8d8d8', fontFamily: 'monomaniac-one' };
    const missText = this.add.text(200, 200, 'Miss: ', labelStyle).setOrigin(1, 1).setAlpha(0);
    missText.depth = 10;
    const resultStyle = { fontSize: '60px', color: '#ffffff', fontFamily: 'monomaniac-one' };
    const missResultText = this.add.text(200, 205, mc, resultStyle).setOrigin(0, 1).setAlpha(0);
    missResultText.depth = 10;
    const timeText = this.add.text(200, 300, 'Time: ', labelStyle).setOrigin(1, 1).setAlpha(0);
    timeText.depth = 10;
    const timeResultText = this.add.text(200, 306, t.toFixed(1) +' sec / ' + String(data.nType) + ' strokes', resultStyle).setOrigin(0, 1).setAlpha(0);
    timeResultText.depth = 10;
    this.results.push(missResultText);
    this.results.push(timeResultText);

    let xt = '';
    let score = 0;
    let tpc = (data.time - 1000) / data.stage!.getCost();
    if (data.missCount == 0 && tpc < 150) {
      score = 1;
      if (tpc < 60) {
        score = 2;
        if (stage.nRun() > 0) {
          xt = ' X' + String(stage.nRun() + 1);
        }
      }
    }
    const hasNext = stage.nextStage(score);
    const evalResultImage = this.add.image(400, 450, 'score' + String(score)).setScale(0.7).setAlpha(0);
    const xText = this.add.text(650, 400, xt, { fontSize: '200px', color: '#ffffff', fontFamily: 'monomaniac-one' }).setOrigin(0.5).setAlpha(0);
    this.results.push(xText);

    const nextListener = (_: any) => {
      if (hasNext) {
        this.scene.transition({target: 'StageScene', duration: 300});
      } else {
        this.game.scene.stop('StageScene');
        this.scene.transition({target: 'LevelScoreScene', duration: 300, data: {stage: stage}});
      }
    };

    this.time.delayedCall(1000, () => {
      this.sound.stopAll();
      this.tweens.add({
        targets: missResultText,
        alpha: 1,
        duration: 200,
      });
    });

    this.time.delayedCall(1200, () => {
      this.tweens.add({
        targets: timeResultText,
        alpha: 1,
        duration: 200,
      });
    });

    this.time.delayedCall(1600, () => {
      this.tweens.add({
        targets: [evalResultImage, xText],
        alpha: 1,
        duration: 200,
      });
      this.sound.playAudioSprite('audio', 'score' + String(score));
      if (score == 2) {
        this.tweens.add({
          targets: evalResultImage,
          x: 400,
          y: 300,
          scale: 2,
          duration: 200,
        });
      }
      this.ibr!.create();
      const nb = this.ibr!.addButton(['スペースキーで', '次に進む'], nextListener);
      this.ibr!.fadeInRegion();
      this.input.keyboard!.on('keydown', (ke: KeyboardEvent) => {
        if (ke.code !== null) {
          if (ke.code == 'Space') {
            nextListener(nb);
          }
        }
      });
    });

    this.tweens.add({
      targets: [background, titleText, missText, timeText],
      alpha: 1,
      duration: 200,
    });

    if (data.missCount == 0) {
      const sc = Cookies.get(stage.getLastHashStr());
      if (sc !== undefined && !Number.isNaN(Number(sc))) {
        const pb = Number(sc);
        if (data.time >= pb) {
          return;
        }
      }
      Cookies.set(stage.getLastHashStr(), String(data.time), { sameSite: 'strict' });
      const pbText = this.add.text(400, 400, 'New Personal Best!', resultStyle).setOrigin(0.5, 1).setAlpha(0);
      pbText.depth = 10;
      this.results.push(pbText);
      this.time.delayedCall(1400, () => {
        this.tweens.add({
          targets: pbText,
          alpha: 1,
          duration: 200,
        });
      });
  
    }
  }
}


class LevelScoreScene extends ScoreSceneBase {
  constructor(){
    super('LevelScoreScene');
  }

  create(data: any) {
    this.sound.stopAll();
    const background = this.add.rectangle(0, 0, Number(this.game.config.width), Number(this.game.config.height), 0x000000).setOrigin(0);
    background.setAlpha(0);

    const stage: Stage = data.stage;
    const scoreN = stage.tally();
    const resultStr = (n: number) => {
      const ns = scoreN[n];
      if (ns == 0) {
        return '-';
      } else if (ns == stage.getNStage()) {
        return 'All Stages!'
      }
      return (String(ns) + ' Stage') + (ns == 1 ? '' : 's');
    }
    const labelText = (msg: string, y: number) => {
      return this.add.text(400, y, msg, { fontSize: '40px', color: '#f8f8f8', fontFamily: 'monomaniac-one' }).setOrigin(1, 1).setAlpha(0);
    }
    const resultText = (msg: string, y: number) => {
      return this.add.text(400, y, msg, { fontSize: '60px', color: '#ffffff', fontFamily: 'monomaniac-one' }).setOrigin(0, 1).setAlpha(0);
    }
    const titleText = this.add.text(400, 50, 'Level Score', { fontSize: '80px', color: '#f8f8f8', fontFamily: 'monomaniac-one' }).setOrigin(0.5).setAlpha(0);
    const okText = labelText('クリアおめでとう: ', 180);
    const okResultText = resultText(resultStr(0), 185);
    let levelScore = 2;
    if (scoreN[0] > 0) {
      this.results.push(okResultText);
      levelScore = 0;
    }
    const fineText = labelText('よくできました: ', 260);
    const fineResultText = resultText(resultStr(1), 265);
    if (scoreN[1] > 0) {
      this.results.push(fineResultText);
      levelScore = Math.min(levelScore, 1);
    }
    const greatText = labelText('たいへんよくできました: ', 340);
    const greatResultText = resultText(resultStr(2), 345);
    if (scoreN[2] > 0) {
      this.results.push(greatResultText);
    }
    const s3 = scoreN[3];
    let rs = s3 == 0 ? '' : ('Max X' + String(s3));
    if (s3 == stage.getNStage()) {
      rs = 'Sweep!';
    }

    const streakResultText = resultText(rs, 395);
    if (scoreN[3] > 0) {
      this.results.push(streakResultText);
    }

    const evalResultImage = this.add.image(400, 500, 'score' + String(levelScore)).setScale(0.5).setAlpha(0);

    let maxLevelScore = levelScore;
    const sc = Cookies.get(stage.level);
    if (sc !== undefined && !Number.isNaN(Number(sc))) {
      maxLevelScore = Math.max(Math.min(Number(sc), 2), maxLevelScore);
    }
    Cookies.set(stage.level, String(maxLevelScore), { sameSite: 'strict' });

    this.tweens.add({
      targets: [background, titleText, okText, fineText, greatText],
      alpha: 1,
      duration: 200,
    });

    this.time.delayedCall(1000, () => {
      this.tweens.add({
        targets: okResultText,
        alpha: 1,
        duration: 200,
      });
    });

    this.time.delayedCall(1200, () => {
      this.tweens.add({
        targets: fineResultText,
        alpha: 1,
        duration: 200,
      });
    });

    this.time.delayedCall(1400, () => {
      this.tweens.add({
        targets: [greatResultText, streakResultText],
        alpha: 1,
        duration: 200,
      });
    });

    this.time.delayedCall(3000, () => {
      let k = 'score' + String(levelScore);
      if (levelScore == 2) {
        k = 'levelScore2';
      }
      this.sound.playAudioSprite('audio', k);
      this.tweens.add({
        targets: evalResultImage,
        alpha: 1,
        duration: 200,
      });
      if (levelScore == 2) {
        this.tweens.add({
          targets: evalResultImage,
          x: 400,
          y: 300,
          scale: 2,
          duration: 3000,
        });
      }
    });

    const buttonDelay = 3000 + ((levelScore == 2) ? 3000 : 0);
    this.time.delayedCall(buttonDelay, () => {
      this.ibr!.create();
      this.ibr!.addButton(['最初に', '戻る'], (_bb: BasicButton)  => {
        this.scene.transition({
          target: 'StartScene',
          duration: 300
        });
      });
      this.ibr!.fadeInRegion();
    });
  }
}

const scale: Phaser.Types.Core.ScaleConfig = {
  mode: Phaser.Scale.FIT,
  autoCenter: Phaser.Scale.CENTER_BOTH,
  width: 800,
  height: 600
};

const config: Phaser.Types.Core.GameConfig = {
	type: Phaser.AUTO,
  title: 'RISCかな配列練習器',
	scale: scale,
	scene: [StartScene, StageScene, WhatIsThisScene, StageScoreScene, LevelScoreScene],
};

new Phaser.Game(config);
